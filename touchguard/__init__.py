"""TouchGuard AI - Real-Time Face-Touch Detection & Voice Warning System.

A modular computer-vision application that monitors the user's webcam,
detects when a hand comes close to or touches their face, and immediately
issues an audible warning.
"""

from __future__ import annotations

__version__ = "2.0.0"
__title__ = "TouchGuard AI - Real-Time Face-Touch Detection & Voice Warning System"

from touchguard.datatypes import FaceBox, HandLandmarks, TouchStatus
from touchguard.pipeline import FrameAnalysis, TouchGuardPipeline

__all__ = [
    "__version__",
    "__title__",
    "FaceBox",
    "HandLandmarks",
    "TouchStatus",
    "FrameAnalysis",
    "TouchGuardPipeline",
]