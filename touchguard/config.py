"""Application configuration for TouchGuard AI.

Configuration sources (highest priority first):

1. A path passed directly to :func:`load_settings`.
2. The ``TOUCHGUARD_CONFIG`` environment variable.
3. ``<project_root>/config/settings.yaml`` (the default file).

When a default file is missing it is created from the built-in defaults. If
PyYAML is unavailable, configuration files cannot be parsed and a
:class:`ConfigurationError` is raised.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from touchguard.exceptions import ConfigurationError

try:  # PyYAML is required to read configuration files.
    import yaml
except ImportError:  # pragma: no cover - only when PyYAML is not installed
    yaml = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class FaceConfig:
    models_dir: str = "assets/models"
    model: str = "haarcascade_frontalface_default.xml"
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: int = 80
    max_detect_width: int = 640

    @property
    def model_path(self) -> Path:
        base = Path(self.models_dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base / self.model


@dataclass
class HandConfig:
    models_dir: str = "models"
    model: str = "hand_landmarker.task"
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    @property
    def model_path(self) -> Path:
        base = Path(self.models_dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base / self.model


@dataclass
class TouchConfig:
    # Distance thresholds (pixels) from the face rectangle. A fingertip closer
    # than `touch_distance_threshold` is a confirmed touch; a fingertip between
    # the two thresholds means the hand is approaching the face. These are
    # relative to a 640x480 frame - scale them for other resolutions.
    approach_distance_threshold: int = 170
    touch_distance_threshold: int = 90
    # Hysteresis: the hand must move further away than
    # ``touch_distance_threshold + release_margin`` pixels before the system
    # re-arms for a new touch event. The default release distance is therefore
    # 90 + 30 = 120 px, so a hand hovering at the touch boundary cannot produce
    # repeated warnings while still touching.
    release_margin: int = 30
    # Temporal confirmation: the touch must persist this many consecutive
    # frames before a warning fires (debounce against single-frame jitter).
    persist_frames: int = 3
    # Re-arm: this many consecutive frames fully outside the release margin
    # must pass before the next touch can trigger a new warning.
    clean_frames: int = 3
    # Count the wrist as a "touching" landmark (False avoids false positives).
    use_wrist: bool = False


@dataclass
class VoiceConfig:
    enabled: bool = True
    # Spoken warning text for every NEW face-touch event. The "{count}"
    # placeholder is replaced with the running touch total, so each separate
    # touch announces its number: "Warning. You touched your face. Touch
    # number 1." / "... number {count}." Messages without the placeholder are
    # spoken verbatim (same text every event).
    main_warning: str = "Warning. You touched your face. Touch number {count}."
    # Minimum interval between spoken warnings (seconds). 0 (default) disables
    # the cooldown so a rapid touch -> remove -> touch cycle is never silently
    # dropped, and every separate touch event gets its own voice warning.
    min_interval_seconds: float = 0.0
    # While the hand *keeps* touching the face, re-announce the warning every
    # this many seconds. 0 (default) speaks exactly once per touch event - a
    # continuous touch is ONE event and never floods warnings. Set > 0 to
    # repeat the reminder until the hand is removed.
    repeat_interval_seconds: float = 0.0
    rate: int = 165
    volume: float = 1.0
    fallback: str = "beep"


@dataclass
class UIConfig:
    window_title: str = "TouchGuard AI"
    show_landmarks: bool = True
    show_fps: bool = True
    show_regions: bool = True


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    touch: TouchConfig = field(default_factory=TouchConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def default_settings() -> Settings:
    return Settings()


def _coerce(target_type: type, value: Any) -> Any:
    if isinstance(value, target_type):
        return value
    if target_type is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return value


def _from_mapping(cls: type, data: Any) -> Any:
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
    kwargs: dict[str, Any] = {}
    for name, field_ in fields.items():
        target_type = hints.get(name, field_.type)
        raw = data.get(name) if isinstance(data, dict) else None
        if raw is None:
            if field_.default is not dataclasses.MISSING:
                value = field_.default
            elif field_.default_factory is not dataclasses.MISSING:
                value = field_.default_factory()
            else:
                value = None
        elif dataclasses.is_dataclass(target_type):
            value = _from_mapping(target_type, raw)
        else:
            value = _coerce(target_type, raw)
        kwargs[name] = value
    return cls(**kwargs)


def _settings_to_dict(settings: Settings) -> dict[str, Any]:
    return dataclasses.asdict(settings)


def _write_default_config(path: Path) -> None:
    if yaml is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(_settings_to_dict(default_settings()), sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Created default configuration file at %s", path)
    except OSError as exc:  # pragma: no cover - depends on environment
        logger.warning("Could not write default config file: %s", exc)


def _resolve_config_path(path: str | os.PathLike | None) -> Path:
    if path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    env_path = os.environ.get("TOUCHGUARD_CONFIG")
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return DEFAULT_CONFIG_PATH


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load application settings from a YAML file (or built-in defaults)."""
    config_path = _resolve_config_path(path)
    data: dict[str, Any] = {}

    if config_path.exists():
        if yaml is None:
            raise ConfigurationError(
                "PyYAML is required to read the configuration file but is not "
                "installed. Install it with: pip install PyYAML"
            )
        try:
            parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid YAML configuration file '{config_path}': {exc}"
            ) from exc
        data = parsed if isinstance(parsed, dict) else {}
        logger.info("Loaded configuration from %s", config_path)
    elif config_path == DEFAULT_CONFIG_PATH:
        _write_default_config(config_path)
    else:
        logger.warning(
            "Configuration file %s not found; using built-in defaults.", config_path
        )

    return _from_mapping(Settings, data)