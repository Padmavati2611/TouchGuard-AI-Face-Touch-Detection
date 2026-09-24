"""Tests for datatypes and configuration loading."""

from __future__ import annotations

import pytest

from touchguard.config import (
    DEFAULT_CONFIG_PATH,
    FaceConfig,
    HandConfig,
    Settings,
    load_settings,
    default_settings,
)
from touchguard.datatypes import FINGERTIP_INDICES, WRIST_INDEX, FaceBox, TouchStatus
from touchguard.exceptions import ConfigurationError
from tests.conftest import face_box_at


def test_facebox_normalizes_corners() -> None:
    box = FaceBox(300.0, 360.0, 100.0, 120.0)
    assert box.x1 == 100.0
    assert box.y1 == 120.0
    assert box.x2 == 300.0
    assert box.y2 == 360.0


def test_facebox_properties() -> None:
    box = face_box_at()
    assert box.width == 200.0
    assert box.height == 240.0
    assert box.center == (200.0, 240.0)
    assert box.to_int_tuple() == (100, 120, 300, 360)


def test_facebox_distance_and_regions() -> None:
    box = face_box_at()
    assert box.distance_to_point(200.0, 240.0) == 0.0  # inside -> 0
    assert box.distance_to_point(400.0, 240.0) == 100.0  # right of the face
    assert box.distance_to_point(100.0, 500.0) == 140.0  # below the face
    regions = {region.name for region in box.regions()}
    assert regions == {"forehead", "nose", "left_cheek", "right_cheek", "mouth_chin"}


def test_touch_status_labels() -> None:
    assert TouchStatus.NO_FACE_DETECTED.value == "NO FACE DETECTED"
    assert TouchStatus.SAFE.value == "SAFE – HAND AWAY FROM FACE"
    assert TouchStatus.HAND_APPROACHING.value == "HAND APPROACHING FACE"
    assert TouchStatus.FACE_TOUCH_DETECTED.value == "⚠ FACE TOUCH DETECTED"


def test_fingertip_constants() -> None:
    assert len(FINGERTIP_INDICES) == 5
    assert all(i in range(21) for i in FINGERTIP_INDICES)
    assert WRIST_INDEX == 0


def test_default_settings() -> None:
    settings = default_settings()
    assert settings.camera.index == 0
    assert settings.camera.width == 640
    assert settings.camera.height == 480
    assert settings.face.max_detect_width == 640
    assert settings.touch.approach_distance_threshold == 170
    assert settings.touch.touch_distance_threshold == 90
    assert settings.touch.release_margin == 30
    assert settings.touch.persist_frames >= 1
    assert settings.voice.main_warning == (
        "Warning. You touched your face. Touch number {count}."
    )
    assert settings.voice.repeat_interval_seconds == 0.0
    assert settings.voice.min_interval_seconds == 0.0
    assert bool(settings.voice.enabled)


def test_hand_model_path_resolves_under_project_root() -> None:
    config = HandConfig(models_dir="models", model="hand_landmarker.task")
    path = config.model_path
    assert str(path).endswith("hand_landmarker.task")
    assert "models" in str(path)


def test_load_settings_defaults_when_file_missing(tmp_path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")
    assert isinstance(settings, Settings)
    assert settings.camera.index == 0


def test_load_settings_overrides_from_yaml(tmp_path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "camera:\n  index: 2\n"
        "touch:\n  approach_distance_threshold: 150\n  touch_distance_threshold: 80\n"
        "  release_margin: 30\n"
        "voice:\n  min_interval_seconds: 3.0\n",
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.camera.index == 2
    assert settings.touch.approach_distance_threshold == 150
    assert settings.touch.touch_distance_threshold == 80
    assert settings.touch.release_margin == 30
    assert settings.voice.min_interval_seconds == 3.0
    assert settings.face.scale_factor == 1.1


def test_load_settings_boolean_and_float_coercion(tmp_path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "touch:\n  use_wrist: true\n"
        "voice:\n  enabled: false\n  volume: 0.5\n",
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.touch.use_wrist is True
    assert settings.voice.enabled is False
    assert settings.voice.volume == 0.5


def test_load_settings_invalid_yaml_raises(tmp_path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("\tthis: is\n\tinvalid-indent\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(cfg)


def test_model_path_resolves_under_project_root() -> None:
    config = FaceConfig(models_dir="assets/models", model="haarcascade.xml")
    path = config.model_path
    assert str(path).endswith("haarcascade.xml")
    assert "assets" in str(path)


def test_default_config_file_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()