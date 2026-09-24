"""Tests for FaceDetector incl. missing/corrupt model handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from touchguard import face_detector as face_detector_module
from touchguard.config import FaceConfig
from touchguard.datatypes import FaceBox
from touchguard.face_detector import FaceDetector


class _FakeCascade:
    def empty(self) -> bool:
        return False

    @staticmethod
    def detectMultiScale(_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)):
        return [(10, 20, 30, 40)]


class _EmptyCascade(_FakeCascade):
    @staticmethod
    def detectMultiScale(*_args, **_kwargs):
        return []


class _CorruptCascade(_FakeCascade):
    def empty(self) -> bool:
        return True


class _FakeCV2:
    COLOR_BGR2GRAY = 6

    def __init__(self, cascade_factory=_FakeCascade) -> None:
        self._cascade_factory = cascade_factory

    def cvtColor(self, frame, code):
        return frame

    def equalizeHist(self, frame):
        return frame

    def CascadeClassifier(self, path):
        return self._cascade_factory()


def _make_config(tmp_path, model="face.xml") -> FaceConfig:
    return FaceConfig(models_dir=str(tmp_path), model=model)


def test_missing_model_degrades_gracefully(tmp_path, monkeypatch, blank_frame) -> None:
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: False)
    detector = FaceDetector(_make_config(tmp_path))
    assert detector.available is False
    assert detector.unavailable_reason
    assert detector.detect(blank_frame) == []


def test_valid_model_detects_faces(tmp_path, monkeypatch, blank_frame) -> None:
    (tmp_path / "face.xml").write_bytes(b"<model/>")
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: True)
    detector = FaceDetector(_make_config(tmp_path))
    detector._cv2 = _FakeCV2()
    boxes = detector.detect(blank_frame)
    assert len(boxes) == 1
    assert isinstance(boxes[0], FaceBox)
    assert boxes[0] == FaceBox(10.0, 20.0, 40.0, 60.0)


def test_no_faces_returns_empty(tmp_path, monkeypatch, blank_frame) -> None:
    (tmp_path / "face.xml").write_bytes(b"<model/>")
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: True)
    detector = FaceDetector(_make_config(tmp_path))
    detector._cv2 = _FakeCV2(_EmptyCascade)
    assert detector.detect(blank_frame) == []


def test_corrupt_model_sets_unavailable(tmp_path, monkeypatch) -> None:
    (tmp_path / "face.xml").write_bytes(b"garbage")
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: True)
    detector = FaceDetector(_make_config(tmp_path))
    detector._cv2 = _FakeCV2(_CorruptCascade)
    assert detector.available is False
    assert detector.unavailable_reason
    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_non_ascii_path_falls_back_to_ascii_copy(tmp_path, monkeypatch, blank_frame) -> None:
    unicode_dir = tmp_path / "TouchGuard \u2013 AI"
    unicode_dir.mkdir()
    (unicode_dir / "cascade.xml").write_bytes(b"<model/>")
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: True)

    ascii_dir = tmp_path / "ascii"
    ascii_dir.mkdir()
    fallback = ascii_dir / "cascade.xml"
    fallback.write_bytes(b"<model/>")

    calls: list[str] = []

    class _UnicodeAwareCV(_FakeCV2):
        def CascadeClassifier(self, path: str):
            calls.append(path)
            try:
                path.encode("ascii")
            except UnicodeEncodeError:
                return _CorruptCascade()  # emulate OpenCV's Windows path failure
            return _FakeCascade()

    config = FaceConfig(models_dir=str(unicode_dir), model="cascade.xml")
    detector = FaceDetector(config)
    detector._cv2 = _UnicodeAwareCV()
    detector._ascii_fallback_path = lambda _p: fallback

    assert detector.available is True
    boxes = detector.detect(blank_frame)
    assert len(boxes) == 1
    assert str(fallback) in calls


class _ResizeRecordingCV(_FakeCV2):
    INTER_AREA = 3

    def __init__(self, cascade_factory=_FakeCascade) -> None:
        super().__init__(cascade_factory)
        self.resize_calls: list[tuple[int, int]] = []

    def resize(self, image, size, interpolation=None):
        self.resize_calls.append(tuple(size))
        return image


def test_wide_frames_are_downscaled_and_boxes_scaled_back(tmp_path, monkeypatch) -> None:
    (tmp_path / "face.xml").write_bytes(b"<model/>")
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: True)

    config = FaceConfig(models_dir=str(tmp_path), model="face.xml", max_detect_width=640)
    detector = FaceDetector(config)
    detector._cv2 = _ResizeRecordingCV()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    boxes = detector.detect(frame)

    assert detector._cv2.resize_calls == [(640, 360)]
    assert len(boxes) == 1
    assert boxes[0] == FaceBox(20.0, 40.0, 80.0, 120.0)


def test_small_frames_are_not_downscaled(tmp_path, monkeypatch) -> None:
    (tmp_path / "face.xml").write_bytes(b"<model/>")
    monkeypatch.setattr(face_detector_module, "ensure_model_exists", lambda *a, **k: True)

    detector = FaceDetector(_make_config(tmp_path))
    detector._cv2 = _ResizeRecordingCV()

    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    boxes = detector.detect(frame)

    assert detector._cv2.resize_calls == []
    assert len(boxes) == 1