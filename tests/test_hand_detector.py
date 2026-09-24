"""Tests for HandDetector incl. graceful degradation without MediaPipe."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from touchguard.config import HandConfig
from touchguard.datatypes import HandLandmarks
from touchguard.hand_detector import HandDetector


class _LM:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _FakeImage:
    def __init__(self, data) -> None:
        self.data = data


class _FakeMP:
    class ImageFormat:
        SRGB = 1

    @staticmethod
    def Image(image_format, data) -> _FakeImage:
        return _FakeImage(data)


def _fake_landmarker(label: str | None = "Right", with_hand: bool = True):
    result = SimpleNamespace(
        hand_landmarks=(
            [[_LM(i / 20.0, i / 20.0) for i in range(21)]] if with_hand else []
        ),
        handedness=None
        if label is None
        else [[SimpleNamespace(label=label)]],
    )

    class _Landmarker:
        def __init__(self) -> None:
            self.closed = False

        def detect_for_video(self, _image, timestamp_ms):
            return result

        def close(self) -> None:
            self.closed = True

    return _Landmarker()


def _prepare_detector(landmarker) -> HandDetector:
    detector = HandDetector(HandConfig(num_hands=2))
    detector._landmarker = landmarker
    detector._mp = _FakeMP
    detector._np = np
    detector._cv2 = SimpleNamespace(
        COLOR_BGR2RGB=4, cvtColor=lambda frame, code: frame
    )
    return detector


def test_landmarks_converted_to_pixel_coordinates(blank_frame) -> None:
    detector = _prepare_detector(_fake_landmarker(label="Right"))
    hands = detector.detect(blank_frame)  # 240 x 320
    assert len(hands) == 1
    hand = hands[0]
    assert isinstance(hand, HandLandmarks)
    assert len(hand.points) == 21
    x, y = hand.points[8]
    assert x == pytest.approx(320.0 * (8 / 20.0))
    assert y == pytest.approx(240.0 * (8 / 20.0))
    assert hand.label == "Right"


def test_hand_label_defaults_when_handedness_missing(blank_frame) -> None:
    detector = _prepare_detector(_fake_landmarker(label=None))
    hands = detector.detect(blank_frame)
    assert len(hands) == 1
    assert hands[0].label == "hand"


def test_no_hands_returns_empty(blank_frame) -> None:
    detector = _prepare_detector(_fake_landmarker(with_hand=False))
    assert detector.detect(blank_frame) == []


def test_tracking_timestamps_are_monotonic(blank_frame) -> None:
    detector = _prepare_detector(_fake_landmarker())
    detector.detect(blank_frame)
    first_ts = detector._tracking_ts_ms
    detector.detect(blank_frame)
    assert detector._tracking_ts_ms > first_ts


def test_degrades_gracefully_when_unavailable(blank_frame, monkeypatch) -> None:
    detector = HandDetector()
    monkeypatch.setattr(detector, "_ensure_landmarker", lambda: None)
    assert detector.available is False
    assert detector.detect(blank_frame) == []


def test_close_releases_landmarker() -> None:
    landmarker = _fake_landmarker()
    detector = _prepare_detector(landmarker)
    detector.close()
    assert landmarker.closed is True
    assert detector._landmarker is None