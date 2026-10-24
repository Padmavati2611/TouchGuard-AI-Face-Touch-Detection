"""Face detection built on OpenCV Haar cascades.

The cascade model is resolved from :attr:`touchguard.config.FaceConfig.model_path`.
If OpenCV cannot load the model from the project path, the detector creates an
ASCII-only temporary copy and loads the cascade from that safe path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from touchguard.config import FaceConfig
from touchguard.datatypes import FaceBox
from touchguard.utils import DEFAULT_CASCADE_URL, ensure_model_exists

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect faces in BGR frames using OpenCV Haar cascades."""

    def __init__(self, face_config: FaceConfig | None = None) -> None:
        self._config = face_config or FaceConfig()
        self._cv2: Any = None
        self._cascade: Any = None
        self._model_checked = False
        self._unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Return True only when the Haar cascade is actually loaded."""
        return self._load_cascade() is not None

    @property
    def unavailable_reason(self) -> str | None:
        """Return the reason the detector is unavailable, if any."""
        return self._unavailable_reason

    def _get_cv2(self) -> Any:
        """Load OpenCV lazily."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2

        return self._cv2

    def _resolve_model(self) -> Path:
        """Resolve the configured Haar cascade model path."""
        path = self._config.model_path

        if self._model_checked:
            return path

        self._model_checked = True

        if ensure_model_exists(path, DEFAULT_CASCADE_URL):
            logger.info("Face detection model ready at %s", path)
        else:
            self._unavailable_reason = (
                f"Face detection model '{path}' could not be found "
                "or downloaded; face detection is disabled."
            )
            logger.warning(self._unavailable_reason)

        return path

    def _load_cascade(self) -> Any:
        """Load the Haar cascade safely.

        First tries the configured path. If Windows/OpenCV cannot read the
        path because the project folder contains a non-ASCII character such
        as an en-dash, an ASCII-only temporary copy is created and loaded.
        """

        if self._cascade is not None:
            return self._cascade

        try:
            cv = self._get_cv2()
            model_path = self._resolve_model()

            # ---------------------------------------------------------
            # First attempt: load directly from the configured path.
            # ---------------------------------------------------------
            candidate = cv.CascadeClassifier(str(model_path))

            try:
                empty = bool(candidate.empty())
            except AttributeError:
                empty = False

            if not empty:
                logger.info(
                    "Loaded face detection model from '%s'.",
                    model_path,
                )

                self._cascade = candidate
                self._unavailable_reason = None
                return self._cascade

            # ---------------------------------------------------------
            # Second attempt: use an ASCII-only temporary path.
            # This handles Windows paths containing characters such as
            # the "–" in:
            #
            # TouchGuard AI – Real-Time Face-Touch Detection...
            # ---------------------------------------------------------
            logger.warning(
                "OpenCV could not load the cascade from '%s'. "
                "Trying an ASCII-only fallback path.",
                model_path,
            )

            fallback_path = self._ascii_fallback_path(model_path)

            if not fallback_path.exists():
                self._unavailable_reason = (
                    f"Face detection model '{model_path}' exists, "
                    "but the ASCII fallback copy could not be created."
                )
                logger.warning(self._unavailable_reason)

                self._cascade = None
                return None

            fallback = cv.CascadeClassifier(str(fallback_path))

            try:
                fallback_empty = bool(fallback.empty())
            except AttributeError:
                fallback_empty = False

            if not fallback_empty:
                logger.info(
                    "Loaded face detection model from ASCII fallback path '%s'.",
                    fallback_path,
                )

                self._cascade = fallback
                self._unavailable_reason = None
                return self._cascade

            # ---------------------------------------------------------
            # Both attempts failed.
            # ---------------------------------------------------------
            self._unavailable_reason = (
                f"Model file at '{model_path}' could not be loaded "
                "by OpenCV from either the original path or the "
                f"ASCII fallback path '{fallback_path}'."
            )

            logger.warning(self._unavailable_reason)

            self._cascade = None
            return None

        except Exception as exc:
            self._unavailable_reason = (
                f"Face detector initialisation failed: {exc}"
            )

            logger.exception(self._unavailable_reason)

            self._cascade = None
            return None

    @staticmethod
    def _ascii_fallback_path(model_path: Path) -> Path:
        """Create an ASCII-only temporary copy of the cascade model."""

        import shutil
        import tempfile

        target = (
            Path(tempfile.gettempdir())
            / "touchguard"
            / model_path.name
        )

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            source_size = model_path.stat().st_size

            if (
                not target.exists()
                or target.stat().st_size != source_size
            ):
                shutil.copyfile(model_path, target)

        except OSError as exc:
            logger.warning(
                "Could not create ASCII fallback copy: %s",
                exc,
            )

        return target

    def _detect_at_scale(
        self,
        cv: Any,
        cascade: Any,
        gray: Any,
    ) -> list[tuple[int, int, int, int]]:
        """Run Haar detection, downscaling very wide frames for speed."""

        height, width = gray.shape[:2]

        max_width = self._config.max_detect_width

        if max_width and width > max_width:
            scale = max_width / float(width)
        else:
            scale = 1.0

        if scale < 1.0:
            small = cv.resize(
                gray,
                (
                    int(round(width * scale)),
                    int(round(height * scale)),
                ),
                interpolation=cv.INTER_AREA,
            )
        else:
            small = gray
            scale = 1.0

        min_size = max(
            2,
            int(round(self._config.min_face_size * scale)),
        )

        raw = cascade.detectMultiScale(
            small,
            scaleFactor=self._config.scale_factor,
            minNeighbors=self._config.min_neighbors,
            minSize=(min_size, min_size),
        )

        if scale >= 1.0:
            return [
                (int(x), int(y), int(w), int(h))
                for x, y, w, h in raw
            ]

        inverse_scale = 1.0 / scale

        return [
            (
                int(round(x * inverse_scale)),
                int(round(y * inverse_scale)),
                int(round(w * inverse_scale)),
                int(round(h * inverse_scale)),
            )
            for x, y, w, h in raw
        ]

    def detect(self, frame: Any) -> list[FaceBox]:
        """Detect faces in a BGR frame."""

        cascade = self._load_cascade()

        if cascade is None:
            return []

        try:
            cv = self._get_cv2()

            # Convert camera frame from BGR to grayscale.
            gray = cv.cvtColor(
                frame,
                cv.COLOR_BGR2GRAY,
            )

            # Improve contrast before Haar detection.
            gray = cv.equalizeHist(gray)

            raw = self._detect_at_scale(
                cv,
                cascade,
                gray,
            )

        except Exception as exc:
            logger.warning(
                "Face detection failed on a frame: %s",
                exc,
            )
            return []

        boxes: list[FaceBox] = []

        for x, y, w, h in raw:
            boxes.append(
                FaceBox(
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                )
            )

        return boxes