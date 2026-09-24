"""Tests for application startup and graceful failure handling."""

from __future__ import annotations

import pathlib

from touchguard.config import Settings, load_settings, default_settings
from touchguard.pipeline import TouchGuardPipeline
from tests.conftest import FakeVoice, FailingStream, StubDetector, face_box_at


def test_default_settings_load_at_startup() -> None:
    settings = load_settings()
    assert isinstance(settings, Settings)


def test_pipeline_builds_without_heavy_models() -> None:
    settings = default_settings()
    pipeline = TouchGuardPipeline(
        settings,
        face_detector=StubDetector(faces=[face_box_at()]),
        hand_detector=StubDetector(hands=[]),
        voice=FakeVoice(),
    )
    assert pipeline.model_status
    pipeline.close()


def test_run_app_reports_webcam_failure() -> None:
    import app as app_module

    exit_code = app_module.run_app(default_settings(), stream=FailingStream())
    assert exit_code == 1


def test_run_app_returns_2_when_no_gui(monkeypatch) -> None:
    import app as app_module

    monkeypatch.setattr(app_module, "cv2_gui_available", lambda: False)
    assert app_module.run_app(default_settings(), stream=FailingStream()) == 2


def test_required_repository_files_exist() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in (
        "README.md",
        "requirements.txt",
        "config/settings.yaml",
        "scripts/download_models.py",
    ):
        assert (root / name).exists(), f"required file missing: {name}"
    assert (root / "models").is_dir(), "missing models directory"


def test_app_module_importable() -> None:
    import app

    assert callable(app.run_app)
    assert callable(app.main)