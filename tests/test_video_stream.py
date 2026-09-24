"""Tests for VideoStream incl. webcam failure handling."""

from __future__ import annotations

import numpy as np
import pytest

from touchguard.exceptions import CameraError
from touchguard.video_stream import VideoStream

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


class FakeCapture:
    def __init__(self, opened: bool = True, frames=None) -> None:
        self._opened = opened
        self._frames = list(frames or [(True, FRAME)])
        self.released = False
        self.set_calls: list = []

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._frames:
            return self._frames.pop(0)
        return (False, None)

    def set(self, prop, value) -> bool:
        self.set_calls.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


def stream_with(factory) -> VideoStream:
    return VideoStream(0, 1280, 720, 30, capture_factory=factory)


def test_open_read_release_success() -> None:
    capture = FakeCapture()
    stream = stream_with(lambda i: capture)
    assert stream.open() is stream
    assert stream.is_open is True
    frame = stream.read()
    assert frame is FRAME
    assert (3, 1280) in capture.set_calls
    assert (4, 720) in capture.set_calls
    assert (5, 30) in capture.set_calls
    stream.release()
    assert capture.released is True
    assert stream.is_open is False


def test_open_failure_raises_and_releases() -> None:
    capture = FakeCapture(opened=False)
    stream = stream_with(lambda i: capture)
    with pytest.raises(CameraError):
        stream.open()
    assert capture.released is True
    assert stream.is_open is False


def test_open_failure_when_factory_returns_none() -> None:
    stream = stream_with(lambda i: None)
    with pytest.raises(CameraError):
        stream.open()


def test_read_without_open_raises() -> None:
    stream = stream_with(lambda i: FakeCapture())
    with pytest.raises(CameraError):
        stream.read()


def test_read_end_of_stream_raises() -> None:
    capture = FakeCapture(frames=[(False, None)])
    stream = stream_with(lambda i: capture)
    stream.open()
    with pytest.raises(CameraError):
        stream.read()


def test_context_manager_releases() -> None:
    capture = FakeCapture()
    with stream_with(lambda i: capture) as stream:
        assert stream.is_open is True
    assert capture.released is True
    assert stream.is_open is False


def test_release_is_idempotent() -> None:
    capture = FakeCapture()
    stream = stream_with(lambda i: capture)
    stream.open()
    stream.release()
    stream.release()
    assert capture.released is True