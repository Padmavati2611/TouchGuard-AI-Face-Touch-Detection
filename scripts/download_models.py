"""Standalone helper that pre-downloads the required model files.

Downloads:

* the OpenCV Haar cascade for face detection (into ``assets/models/``), and
* the MediaPipe Hand Landmarker task file (into ``models/``).

Usage::

    python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from touchguard.config import load_settings  # noqa: E402
from touchguard.face_detector import FaceDetector  # noqa: E402
from touchguard.hand_detector import HandDetector  # noqa: E402


def main() -> int:
    settings = load_settings()
    exit_code = 0

    face = FaceDetector(settings.face)
    if face.available:
        print(f"[OK] Face detection model ready at {settings.face.model_path}")
    else:
        print(f"[ERROR] {face.unavailable_reason}", file=sys.stderr)
        exit_code = 1

    hand = HandDetector(settings.hand)
    if hand.available:
        print(f"[OK] Hand landmarker model ready at {settings.hand.model_path}")
    else:
        print(f"[ERROR] {hand.unavailable_reason}", file=sys.stderr)
        exit_code = 1
    hand.close()

    if exit_code == 0:
        print("[OK] All models are ready. Launch the app with: python app.py")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())