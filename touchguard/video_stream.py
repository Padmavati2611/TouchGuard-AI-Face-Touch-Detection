"""Webcam abstraction with graceful open / read / release semantics."""

from __future__ import annotations

import logging
from typing import Any, Callable

from touchguard.exceptions import CameraError

logger = logging.getLogger(__name__)

# OpenCV capture property ids without importing cv2 eagerly.
_CAP_FRAME_WIDTH = 3
_CAP_FRAME_HEIGHT = 4
_CAP_FPS = 5


class VideoStream:
    """Wraps an OpenCV ``VideoCapture`` with safe lifecycle management."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        capture_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self._camera_index = int(camera_index)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._capture_factory = capture_factory or self._default_capture_factory
        self._capture: Any = None

    @staticmethod
    def _default_capture_factory(index: int) -> Any:
        import sys

        import cv2

        if sys.platform.startswith("win"):
            # DirectShow initialises far faster than the auto/MediaFoundation
            # backends on Windows (and preserves the camera's native mode).
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(index)

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def open(self) -> "VideoStream":
        """Open the webcam and apply the configured properties.

        On failure, Windows-specific capture backends (DirectShow and
        MediaFoundation) are tried automatically before giving up.

        Raises :class:`CameraError` when the requested camera cannot be
        opened, including when the camera is already in use.
        """
        if self._capture is not None:
            return self

        capture = self._try_open_capture()
        if capture is None:
            raise CameraError(
                f"Could not open the webcam at index {self._camera_index}. "
                "Check that a webcam is connected and not used by another "
                "application."
            )

        self._apply_properties(capture)
        self._capture = capture
        logger.info(
            "Webcam opened: index=%d requested=%dx%d@%dfps",
            self._camera_index,
            self._width,
            self._height,
            self._fps,
        )
        return self

    def _try_open_capture(self) -> Any:
        factories = [self._capture_factory]
        if self._capture_factory is self._default_capture_factory:
            factories += [
                self._dshow_capture_factory,
                self._msmf_capture_factory,
            ]
        for factory in factories:
            capture = None
            try:
                capture = factory(self._camera_index)
                if capture is not None and bool(capture.isOpened()):
                    return capture
            except Exception:  # noqa: BLE001 - try the next backend
                pass
            if capture is not None:
                try:
                    capture.release()
                except Exception:  # noqa: BLE001 - best-effort
                    pass
        return None

    @staticmethod
    def _dshow_capture_factory(index: int) -> Any:
        import cv2

        return cv2.VideoCapture(index, cv2.CAP_DSHOW)

    @staticmethod
    def _msmf_capture_factory(index: int) -> Any:
        import cv2

        return cv2.VideoCapture(index, cv2.CAP_MSMF)

    def _apply_properties(self, capture: Any) -> None:
        for prop, value in (
            (_CAP_FRAME_WIDTH, self._width),
            (_CAP_FRAME_HEIGHT, self._height),
            (_CAP_FPS, self._fps),
        ):
            try:
                capture.set(prop, value)
            except Exception:  # noqa: BLE001 - properties are best-effort
                pass

    def read(self) -> Any:
        """Read the next frame.

        Raises :class:`CameraError` when the camera is not open or when the
        camera stops producing frames (e.g. it was disconnected).
        """
        if self._capture is None:
            raise CameraError(
                "Cannot read a frame: the camera is not open. Call open() first."
            )
        try:
            ok, frame = self._capture.read()
        except Exception as exc:  # noqa: BLE001 - malformed capture objects
            raise CameraError(f"Failed to read a frame from the webcam: {exc}") from exc
        if not ok or frame is None:
            raise CameraError("The webcam stopped producing frames.")
        return frame

    def release(self) -> None:
        """Release the webcam so other applications can use it."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            self._capture = None
            logger.info("Webcam released.")

    def __enter__(self) -> "VideoStream":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def discover_camera_index(preferred: int = 0, max_index: int = 3) -> int:
    """Return the first working webcam index, raising :class:`CameraError`.

    Probes the preferred index first, then scans ``0..max_index``.
    """
    import cv2

    indices = [preferred] + [i for i in range(max_index + 1) if i != preferred]
    for index in indices:
        try:
            capture = cv2.VideoCapture(index)
            ok = bool(capture.isOpened())
            capture.release()
        except Exception:  # noqa: BLE001 - skip unreadable indices
            ok = False
        if ok:
            return index
    raise CameraError(
        "No working webcam found. Check that a webcam is connected and not "
        "being used by another application."
    )