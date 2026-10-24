"""Shared data types used across TouchGuard AI.

These plain data containers decouple the detection modules from the UI and
from each other, keeping the codebase modular and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

WRIST_INDEX = 0
FINGERTIP_INDICES = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, little

# Face regions that are relevant for face-touch detection. Each entry maps a
# normalised position inside the face box to a named region.
FACE_REGION_SPECS: dict[str, tuple[float, float]] = {
    "forehead": (0.50, 0.18),
    "nose": (0.50, 0.60),
    "left_cheek": (0.20, 0.55),
    "right_cheek": (0.80, 0.55),
    "mouth_chin": (0.50, 0.86),
}

FACE_REGION_LABELS: dict[str, str] = {
    "forehead": "FOREHEAD",
    "nose": "NOSE",
    "left_cheek": "L-CHEEK",
    "right_cheek": "R-CHEEK",
    "mouth_chin": "MOUTH/CHIN",
}


class TouchStatus(str, Enum):
    """Operational status of the monitoring system."""

    NO_FACE_DETECTED = "NO FACE DETECTED"
    SAFE = "SAFE – HAND AWAY FROM FACE"
    HAND_APPROACHING = "HAND APPROACHING FACE"
    FACE_TOUCH_DETECTED = "⚠ FACE TOUCH DETECTED"


@dataclass(frozen=True)
class FaceBox:
    """A detected face region in pixel coordinates (x1 <= x2, y1 <= y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        object.__setattr__(self, "x1", float(x1))
        object.__setattr__(self, "y1", float(y1))
        object.__setattr__(self, "x2", float(x2))
        object.__setattr__(self, "y2", float(y2))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def regions(self) -> list["FaceRegion"]:
        """Return anchor points for the important face regions.

        The anchor positions are expressed as fractions of the face box so the
        regions stay aligned as the user moves closer or farther away.
        """
        regions: list[FaceRegion] = []
        for name, (fx, fy) in FACE_REGION_SPECS.items():
            regions.append(
                FaceRegion(
                    name=name,
                    label=FACE_REGION_LABELS.get(name, name),
                    point=(self.x1 + fx * self.width, self.y1 + fy * self.height),
                )
            )
        return regions

    def distance_to_point(self, px: float, py: float) -> float:
        """Euclidean distance from a point to this face rectangle.

        A point inside the rectangle has distance ``0.0``. The distance grows
        smoothly as the point moves away from the face, which is what the
        approach / touch thresholds are calibrated against.
        """
        dx = max(self.x1 - px, max(0.0, px - self.x2))
        dy = max(self.y1 - py, max(0.0, py - self.y2))
        return math.hypot(dx, dy)

    def nearest_point_to(self, px: float, py: float) -> tuple[float, float]:
        """Clamp ``(px, py)`` to the closest point on the rectangle perimeter."""
        return (
            min(max(px, self.x1), self.x2),
            min(max(py, self.y1), self.y2),
        )

    @classmethod
    def closest_region_to(
        cls, faces: list["FaceBox"], px: float, py: float
    ) -> tuple["FaceRegion", "FaceBox"] | None:
        """Return the nearest face region anchor across the given faces."""
        best: tuple[FaceRegion, FaceBox] | None = None
        best_dist = math.inf
        for face in faces:
            for region in face.regions():
                dist = math.hypot(px - region.point[0], py - region.point[1])
                if dist < best_dist:
                    best_dist = dist
                    best = (region, face)
        return best


@dataclass(frozen=True)
class FaceRegion:
    """Named anchor describing a region of interest on a face."""

    name: str
    label: str
    point: tuple[float, float]


@dataclass
class HandLandmarks:
    """A detected hand with its 21 MediaPipe landmarks in pixel coordinates."""

    label: str = "hand"
    points: list[tuple[float, float]] = field(default_factory=list)

    @property
    def fingertip_points(self) -> list[tuple[float, float]]:
        return [self.points[i] for i in FINGERTIP_INDICES if i < len(self.points)]