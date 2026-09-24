"""Shared fixtures and doubles used across the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from touchguard.datatypes import FINGERTIP_INDICES, FaceBox, HandLandmarks
from touchguard.exceptions import CameraError


def face_box_at(x1: float = 100.0, y1: float = 120.0, x2: float = 300.0, y2: float = 360.0) -> FaceBox:
    return FaceBox(x1, y1, x2, y2)


def hand_with_points(points: list[tuple[int, tuple[float, float]]]) -> HandLandmarks:
    pts: list[tuple[float, float]] = [(0.0, 0.0) for _ in range(21)]
    for index, point in points:
        pts[index] = point
    return HandLandmarks(label="hand", points=pts)


def _hand_at(face: FaceBox, x: float, y: float) -> HandLandmarks:
    """Build a hand whose fingertips all sit at the given pixel position."""
    return hand_with_points([(index, (x, y)) for index in FINGERTIP_INDICES])


def touching_hand(face: FaceBox) -> HandLandmarks:
    """Index fingertip at the face centre: distance 0 -> confirmed touch."""
    return _hand_at(face, face.center_x, face.center_y)


def idle_hand(face: FaceBox) -> HandLandmarks:
    """Fingertips far from the face: distance > approach threshold -> SAFE."""
    return _hand_at(face, face.x2 + 250.0, face.y1 + 50.0)


def approaching_hand(face: FaceBox) -> HandLandmarks:
    """Fingertips between the touch and approach thresholds (~120 px)."""
    return _hand_at(face, face.x2 + 120.0, face.center_y)


def hovering_hand(face: FaceBox) -> HandLandmarks:
    """Fingertips just outside the touch threshold but inside release margin."""
    return _hand_at(face, face.x2 + 110.0, face.center_y)


def released_hand(face: FaceBox) -> HandLandmarks:
    """Fingertips clearly outside the release margin -> re-arms detection."""
    return _hand_at(face, face.x2 + 300.0, face.center_y)


class FakeVoice:
    """Recording stand-in for VoiceWarningService."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.mode = "fake"
        self.available = True
        self.warning_count = 0
        self.messages: list[str | None] = []
        self.stop_called = False

    def announce(self, message: str | None = None) -> bool:
        if not self.enabled:
            return False
        self.messages.append(message)
        self.warning_count += 1
        return True

    @property
    def cooldown_remaining(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_called = True


class StubDetector:
    """Returns a fixed list of detections and counts its calls."""

    def __init__(self, faces: list = (), hands: list = ()) -> None:
        self.faces = list(faces)
        self.hands = list(hands)
        self.calls = 0
        self.close_called = False

    def detect(self, frame):
        self.calls += 1
        return list(self.faces if self.faces else self.hands)

    def close(self) -> None:
        self.close_called = True


class FailingStream:
    def open(self):
        raise CameraError("Webcam not found (test)")


@pytest.fixture
def blank_frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def face() -> FaceBox:
    return face_box_at()