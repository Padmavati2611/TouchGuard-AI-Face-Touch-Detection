"""Face-touch detection logic - distance-based, event-driven state machine."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from touchguard.config import TouchConfig
from touchguard.datatypes import (
    FINGERTIP_INDICES,
    WRIST_INDEX,
    FaceBox,
    HandLandmarks,
    TouchStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class TouchDecision:
    """Result of evaluating one frame against the touch state machine."""

    status: TouchStatus = TouchStatus.SAFE
    touching: bool = False
    should_warn: bool = False
    distance: float = math.inf
    closest_point: tuple[float, float] | None = None
    closest_region_name: str | None = None
    region_point: tuple[float, float] | None = None
    touch_count: int = 0
    touch_streak: int = 0
    clean_streak: int = 0
    touch_active: bool = False


class FaceTouchLogic:
    """Evaluate face/hand geometry and detect separate face-touch events."""

    def __init__(self, touch_config: TouchConfig | None = None) -> None:
        config = touch_config or TouchConfig()

        self._approach = max(
            0.0,
            float(config.approach_distance_threshold),
        )

        self._touch = max(
            0.0,
            float(config.touch_distance_threshold),
        )

        self._margin = max(
            0.0,
            float(config.release_margin),
        )

        self._persist_frames = max(
            1,
            int(config.persist_frames),
        )

        self._clean_frames = max(
            1,
            int(config.clean_frames),
        )

        self._use_wrist = bool(config.use_wrist)

        # Current touch state.
        self._touch_active = False

        # Number of consecutive frames inside the touch zone.
        self._touch_streak = 0

        # Number of consecutive frames outside the release zone.
        self._clean_streak = 0

        # Total number of confirmed face touches.
        self._touch_count = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def touch_count(self) -> int:
        return self._touch_count

    @property
    def touch_active(self) -> bool:
        return self._touch_active

    @property
    def touch_streak(self) -> int:
        return self._touch_streak

    @property
    def clean_streak(self) -> int:
        return self._clean_streak

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Completely reset the touch state."""
        self._touch_active = False
        self._touch_streak = 0
        self._clean_streak = 0
        self._touch_count = 0

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def update(
        self,
        faces: list[FaceBox],
        hands: list[HandLandmarks],
    ) -> TouchDecision:
        """Process one frame and return the current touch decision."""

        # --------------------------------------------------------------
        # No face detected
        # --------------------------------------------------------------
        if not faces:
            self._touch_streak = 0
            self._clean_streak += 1

            # No face means there cannot be an active face-touch event.
            # Re-arm the system after the configured clean period.
            if self._clean_streak >= self._clean_frames:
                self._touch_active = False
                self._clean_streak = 0

            return TouchDecision(
                status=TouchStatus.NO_FACE_DETECTED,
                distance=math.inf,
                touch_count=self._touch_count,
                touch_active=self._touch_active,
                clean_streak=self._clean_streak,
            )

        # --------------------------------------------------------------
        # Find closest hand point to the face
        # --------------------------------------------------------------
        hand_geometry = self._closest_hand_geometry(faces, hands)

        if hand_geometry is None:
            # No hand detected.
            self._touch_streak = 0
            self._handle_release(math.inf)

            return TouchDecision(
                status=TouchStatus.SAFE,
                touching=False,
                distance=math.inf,
                touch_count=self._touch_count,
                touch_streak=0,
                clean_streak=self._clean_streak,
                touch_active=self._touch_active,
            )

        distance = hand_geometry.distance

        # --------------------------------------------------------------
        # HAND INSIDE TOUCH ZONE
        # --------------------------------------------------------------
        if distance <= self._touch:

            # The hand is touching the face.
            self._touch_streak += 1
            self._clean_streak = 0

            should_warn = False

            # Only create a new event if the previous touch was released.
            if (
                not self._touch_active
                and self._touch_streak >= self._persist_frames
            ):
                self._touch_active = True
                self._touch_count += 1
                should_warn = True

                logger.info(
                    "Face touch event confirmed (count=%d).",
                    self._touch_count,
                )

        # --------------------------------------------------------------
        # HAND OUTSIDE TOUCH ZONE
        # --------------------------------------------------------------
        else:
            self._touch_streak = 0

            self._handle_release(distance)

            should_warn = False

        touching = distance <= self._touch
        status = self._status_for_distance(distance)

        return TouchDecision(
            status=status,
            touching=touching,
            should_warn=should_warn,
            distance=distance,
            closest_point=hand_geometry.point,
            closest_region_name=hand_geometry.region_name,
            region_point=hand_geometry.region_point,
            touch_count=self._touch_count,
            touch_streak=self._touch_streak,
            clean_streak=self._clean_streak,
            touch_active=self._touch_active,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _status_for_distance(self, distance: float) -> TouchStatus:
        if distance <= self._touch:
            return TouchStatus.FACE_TOUCH_DETECTED

        if distance <= self._approach:
            return TouchStatus.HAND_APPROACHING

        return TouchStatus.SAFE

    # ------------------------------------------------------------------
    # Release / re-arm
    # ------------------------------------------------------------------

    def _handle_release(self, distance: float) -> None:
        """Re-arm after the hand has clearly moved away from the face."""

        # The hand must move beyond the release boundary.
        release_at = self._touch + self._margin

        if distance > release_at:
            self._clean_streak += 1

            if self._clean_streak >= self._clean_frames:

                if self._touch_active:
                    logger.info(
                        "Face-touch zone released; system re-armed."
                    )

                self._touch_active = False
                self._clean_streak = 0

        else:
            self._clean_streak = 0

    # ------------------------------------------------------------------
    # Landmark selection
    # ------------------------------------------------------------------

    def _point_landmarks(self) -> list[int]:
        """Return landmarks used for face-touch detection."""

        indices = list(FINGERTIP_INDICES)

        if self._use_wrist:
            indices.append(WRIST_INDEX)

        return indices

    # ------------------------------------------------------------------
    # Closest hand / face geometry
    # ------------------------------------------------------------------

    def _closest_hand_geometry(
        self,
        faces: list[FaceBox],
        hands: list[HandLandmarks],
    ) -> Any | None:
        """Find the closest hand landmark to any detected face."""

        best: Any | None = None

        for hand in hands:

            for index in self._point_landmarks():

                if index >= len(hand.points):
                    continue

                point = hand.points[index]

                face = min(
                    faces,
                    key=lambda f: f.distance_to_point(
                        point[0],
                        point[1],
                    ),
                )

                distance = face.distance_to_point(
                    point[0],
                    point[1],
                )

                if best is None or distance < best.distance:

                    region = FaceBox.closest_region_to(
                        [face],
                        point[0],
                        point[1],
                    )

                    best = _HandGeometry(
                        point=point,
                        distance=distance,
                        region_name=(
                            region[0].name
                            if region
                            else None
                        ),
                        region_point=(
                            region[0].point
                            if region
                            else None
                        ),
                    )

        return best


@dataclass
class _HandGeometry:
    point: tuple[float, float]
    distance: float
    region_name: str | None
    region_point: tuple[float, float] | None