"""Custom exceptions raised by TouchGuard AI.

Every exception inherits from :class:`TouchGuardError` so callers can catch
a single base type, while keeping enough granularity for fine-grained
error handling.
"""

from __future__ import annotations


class TouchGuardError(Exception):
    """Base exception for all TouchGuard AI errors."""


class ConfigurationError(TouchGuardError):
    """Raised when the application configuration is missing or invalid."""


class CameraError(TouchGuardError):
    """Raised when the webcam cannot be opened, read or released."""


class ModelError(TouchGuardError):
    """Raised when a required model file is missing or invalid."""


class VoiceError(TouchGuardError):
    """Raised when the voice warning subsystem cannot be initialised."""