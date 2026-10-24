"""Monitoring pipeline - orchestrates one frame of processing.

The pipeline wires together face detection, hand/landmark detection, the
face-touch state machine and the voice warning service, then returns a
fully-annotated :class:`FrameAnalysis` for the UI to display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from touchguard.config import Settings, load_settings
from touchguard.datatypes import FaceBox, HandLandmarks, TouchStatus
from touchguard.face_detector import FaceDetector
from touchguard.face_touch_detector import FaceTouchLogic, TouchDecision
from touchguard.hand_detector import HandDetector
from touchguard.utils import annotate_frame
from touchguard.voice_warning import VoiceWarningService

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """Result of processing a single video frame."""

    frame: Any
    status: TouchStatus = TouchStatus.SAFE
    should_warn: bool = False
    touching: bool = False
    faces: list[FaceBox] = field(default_factory=list)
    hands: list[HandLandmarks] = field(default_factory=list)
    distance: float = float("inf")
    closest_point: tuple[float, float] | None = None
    region_point: tuple[float, float] | None = None
    region_name: str | None = None
    touch_count: int = 0
    fps: float = 0.0
    processing_ms: float = 0.0
    warning_count: int = 0
    cooldown_remaining: float = 0.0
    voice_enabled: bool = True


class TouchGuardPipeline:
    """Composite detector that turns one BGR frame into a :class:`FrameAnalysis`.

    All collaborators can be injected for testing; when omitted they are
    built from the application settings.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        face_detector: FaceDetector | None = None,
        hand_detector: HandDetector | None = None,
        logic: FaceTouchLogic | None = None,
        voice: VoiceWarningService | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.face_detector = face_detector or FaceDetector(self.settings.face)
        self.hand_detector = hand_detector or HandDetector(self.settings.hand)
        self.logic = logic or FaceTouchLogic(self.settings.touch)
        self.voice = voice or VoiceWarningService(self.settings.voice)
        self._smooth_ms = 0.0
        self._fps = 0.0
        self._close_called = False
        self._last_warning_time: float | None = None

    def _update_fps(self, processing_ms: float) -> None:
        """Track a smoothed per-frame processing time.

        The first frames may include expensive model initialisation; when a
        large speed-up is observed the exponential moving average restarts so
        the initialisation cost does not drag the reported FPS down.
        """
        if self._smooth_ms == 0.0:
            self._smooth_ms = max(processing_ms, 1e-3)
        elif processing_ms < self._smooth_ms * 0.5:
            self._smooth_ms = max(processing_ms, 1e-3)
        else:
            self._smooth_ms = self._smooth_ms * 0.9 + processing_ms * 0.1
        self._fps = 1000.0 / self._smooth_ms

    @property
    def model_status(self) -> dict[str, Any]:
        """Human-readable availability of the heavy model backends."""
        return {
            "face": {
                "available": getattr(self.face_detector, "available", True),
                "reason": getattr(self.face_detector, "unavailable_reason", None),
            },
            "hand": {
                "available": getattr(self.hand_detector, "available", True),
                "reason": getattr(self.hand_detector, "unavailable_reason", None),
            },
            "voice": {"mode": getattr(self.voice, "mode", "n/a")},
        }

    def _announce_warning(self, touch_count: int) -> None:
        """Speak the warning with the running face-touch total."""
        try:
            message = self.settings.voice.main_warning.format(count=touch_count)
        except (KeyError, IndexError, ValueError):
            message = self.settings.voice.main_warning
        self.voice.announce(message)
        self._last_warning_time = perf_counter()

    def process(self, frame: Any) -> FrameAnalysis:
        """Run one full detection/warning cycle on ``frame``."""
        start = perf_counter()

        faces = self.face_detector.detect(frame) or []
        hands = self.hand_detector.detect(frame) or []

        decision: TouchDecision = self.logic.update(faces, hands)

        if decision.should_warn:
            self._announce_warning(decision.touch_count)
        elif (
            decision.touching
            and self._last_warning_time is not None
            and self.settings.voice.repeat_interval_seconds > 0
        ):
            # The hand is keeping the face touched: re-remind periodically so
            # the user hears the (repeating) total until they pull away.
            if (perf_counter() - self._last_warning_time) >= self.settings.voice.repeat_interval_seconds:
                self._announce_warning(decision.touch_count)
        elif not decision.touching:
            self._last_warning_time = None

        processing_ms = (perf_counter() - start) * 1000.0
        self._update_fps(processing_ms)

        annotated = annotate_frame(
            frame,
            faces,
            hands,
            decision.status,
            self.settings.ui,
            self._fps,
            decision.touch_count,
            decision.distance,
            decision.closest_point,
            decision.region_point,
            decision.closest_region_name,
        )
        return FrameAnalysis(
            frame=annotated,
            status=decision.status,
            should_warn=decision.should_warn,
            touching=decision.touching,
            faces=faces,
            hands=hands,
            distance=decision.distance,
            closest_point=decision.closest_point,
            region_point=decision.region_point,
            region_name=decision.closest_region_name,
            touch_count=decision.touch_count,
            fps=self._fps,
            processing_ms=processing_ms,
            warning_count=self.voice.warning_count,
            cooldown_remaining=self.voice.cooldown_remaining,
            voice_enabled=self.voice.enabled,
        )

    def reset(self) -> None:
        """Reset event state and the warning cooldown."""
        self.logic.reset()
        self._last_warning_time = None
        self.voice.reset()

    def close(self) -> None:
        """Release background resources (voice thread/engine)."""
        if not self._close_called:
            self._close_called = True
            close_hand_detector = getattr(self.hand_detector, "close", None)
            if close_hand_detector is not None:
                close_hand_detector()
            self.voice.stop()

    def __enter__(self) -> "TouchGuardPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()