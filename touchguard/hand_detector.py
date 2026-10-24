"""Hand / landmark detection built on MediaPipe's Hand Landmarker (Tasks API).

Foundational technology:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

The model file (``models/hand_landmarker.task``) is downloaded automatically on
first use when missing. If the model cannot be obtained, or the MediaPipe Tasks
runtime cannot be initialised, the detector degrades gracefully: it reports
``available == False`` and :meth:`detect` returns an empty list.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from touchguard.config import HandConfig
from touchguard.datatypes import HandLandmarks
from touchguard.utils import HAND_LANDMARKER_MODEL_URL, ensure_model_exists

logger = logging.getLogger(__name__)

# Nominal inter-frame interval used to build strictly increasing timestamps,
# which Video-running-mode asks for.
_FRAME_INTERVAL_MS = 33


class HandDetector:
    """Track hands in a BGR frame and return pixel-space landmarks.

    Parameters
    ----------
    hand_config:
        Optional :class:`HandConfig` controlling the model file and the
        MediaPipe confidence thresholds. Defaults to built-in configuration.
    """

    def __init__(self, hand_config: HandConfig | None = None) -> None:
        self._config = hand_config or HandConfig()
        self._mp: Any = None
        self._cv2: Any = None
        self._np: Any = None
        self._landmarker: Any = None
        self._model_checked = False
        self._tracking_ts_ms = 0
        self._unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self._ensure_landmarker() is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _resolve_model(self) -> Path:
        path = self._config.model_path
        if self._model_checked:
            return path
        self._model_checked = True
        if ensure_model_exists(path, HAND_LANDMARKER_MODEL_URL):
            logger.info("Hand landmarker model ready at %s", path)
        else:
            self._unavailable_reason = (
                f"Hand landmarker model '{path}' could not be found or "
                "downloaded; hand detection is disabled for this session."
            )
            logger.warning(self._unavailable_reason)
        return path

    @staticmethod
    def _ascii_safe_path(model_path: Path) -> Path:
        """Return a model path that is safe to open on Windows.

        MediaPipe's (and OpenCV's) C++ file API cannot open files whose
        absolute path contains non-ASCII characters (e.g. an en-dash in a
        folder name). When that happens the model is copied to an ASCII-only
        temp location and that copy is used instead.
        """
        try:
            model_path.as_posix().encode("ascii")
            return model_path
        except UnicodeEncodeError:
            pass
        import shutil
        import tempfile

        target = Path(tempfile.gettempdir()) / "touchguard" / model_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not target.exists() or target.stat().st_size != model_path.stat().st_size:
                shutil.copyfile(model_path, target)
            logger.info("Using ASCII-safe model path '%s'.", target)
            return target
        except OSError:  # pragma: no cover - best-effort copy
            return model_path

    def _ensure_landmarker(self) -> Any:
        if self._landmarker is not None:
            return self._landmarker
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                model_path = self._ascii_safe_path(self._resolve_model())
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=max(1, self._config.num_hands),
                    min_hand_detection_confidence=self._config.min_detection_confidence,
                    min_hand_presence_confidence=self._config.min_presence_confidence,
                    min_tracking_confidence=self._config.min_tracking_confidence,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
                return self._landmarker
            except Exception as exc:  # pragma: no cover - depends on environment
                last_exc = exc
                if attempt < 3:
                    time.sleep(0.5)
        self._unavailable_reason = (
            f"MediaPipe Hand Landmarker could not be initialised: {last_exc}"
        )
        logger.warning(self._unavailable_reason)
        return None

    def detect(self, frame: Any) -> list[HandLandmarks]:
        """Detect hands in a BGR frame; landmarks are returned in pixels."""
        landmarker = self._ensure_landmarker()
        if landmarker is None:
            return []
        try:
            if self._mp is None:
                import mediapipe as mp

                self._mp = mp
            if self._np is None:
                import numpy as np

                self._np = np
            if self._cv2 is None:
                import cv2

                self._cv2 = cv2
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            rgb = self._np.ascontiguousarray(rgb)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb
            )
            self._tracking_ts_ms += _FRAME_INTERVAL_MS
            results = landmarker.detect_for_video(
                mp_image, timestamp_ms=self._tracking_ts_ms
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Hand detection failed on a frame: %s", exc)
            return []

        height, width = frame.shape[:2]
        landmark_lists = getattr(results, "hand_landmarks", None) or []
        handedness = getattr(results, "handedness", None)
        hands: list[HandLandmarks] = []
        for index, landmarks in enumerate(landmark_lists):
            label = "hand"
            if handedness is not None and index < len(handedness):
                categories = handedness[index]
                if categories and getattr(categories[0], "label", None):
                    label = categories[0].label
            points = [
                (float(landmark.x) * width, float(landmark.y) * height)
                for landmark in landmarks
            ]
            hands.append(HandLandmarks(label=label, points=points))
        return hands

    def close(self) -> None:
        """Release the underlying landmarker instance, if one was created."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # pragma: no cover - best-effort release
                pass
            self._landmarker = None
            self._model_checked = False