"""Tests for utility helpers: model availability and frame annotation."""

from __future__ import annotations

import urllib.error

from touchguard.datatypes import TouchStatus
from touchguard.utils import HAND_CONNECTIONS, annotate_frame, ensure_model_exists
from tests.conftest import face_box_at, idle_hand, touching_hand


def test_existing_model_returns_true(tmp_path) -> None:
    model = tmp_path / "model.xml"
    model.write_bytes(b"<xml/>")
    assert ensure_model_exists(model, url="") is True


def test_empty_existing_model_triggers_download(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.xml"
    model.write_bytes(b"")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @staticmethod
        def read():
            return b"<xml/>"

    monkeypatch.setattr(
        "touchguard.utils.urllib.request.urlopen", lambda *a, **k: _Response()
    )
    assert ensure_model_exists(model, url="https://example.com/model.xml") is True
    assert model.read_bytes() == b"<xml/>"


def test_download_failure_returns_false(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.xml"

    def fail(*a, **k):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("touchguard.utils.urllib.request.urlopen", fail)
    assert ensure_model_exists(model, url="https://example.com/model.xml") is False
    assert not model.exists()


def test_hand_connection_topology() -> None:
    assert len(HAND_CONNECTIONS) == 21
    flattened = [n for pair in HAND_CONNECTIONS for n in pair]
    assert max(flattened) == 20
    assert set(flattened) == set(range(21))


def test_annotate_frame_returns_copy_with_overlay(blank_frame, face) -> None:
    decorated = annotate_frame(
        blank_frame,
        faces=[face],
        hands=[idle_hand(face)],
        status=TouchStatus.SAFE,
        touch_count=0,
        distance=250.0,
        closest_point=idle_hand(face).fingertip_points[1],
        region_point=(200.0, 163.2),
        region_name="forehead",
    )
    assert decorated.shape == blank_frame.shape
    assert not (decorated == blank_frame).all()  # overlays were drawn


def test_annotate_frame_handles_empty_input(blank_frame) -> None:
    decorated = annotate_frame(blank_frame, faces=[], hands=[], status=TouchStatus.SAFE)
    assert decorated.shape == blank_frame.shape


def test_annotate_frame_warning_panel(blank_frame, face) -> None:
    decorated = annotate_frame(
        blank_frame,
        faces=[face],
        hands=[touching_hand(face_box_at())],
        status=TouchStatus.FACE_TOUCH_DETECTED,
        touch_count=3,
        distance=0.0,
        closest_point=face.center,
        region_point=face.regions()[1].point,
        region_name="nose",
    )
    assert decorated.shape == blank_frame.shape
    assert not (decorated == blank_frame).all()  # panel + landmarks drawn