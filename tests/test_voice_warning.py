"""Tests for VoiceWarningService: cooldown, scheduling, degradation, stop."""

from __future__ import annotations

from touchguard.config import VoiceConfig
from touchguard.voice_warning import VoiceWarningService


class FakeEngine:
    def __init__(self) -> None:
        self.say_calls: list[str] = []
        self.run_count = 0
        self.stop_count = 0
        self.props: dict = {}

    def say(self, text: str) -> None:
        self.say_calls.append(text)

    def runAndWait(self) -> None:
        self.run_count += 1

    def setProperty(self, key, value) -> None:  # noqa: ANN001 - generic fixture
        self.props[key] = value

    def stop(self) -> None:
        self.stop_count += 1


def build(engine_factory, config: VoiceConfig | None = None, **kwargs):
    return VoiceWarningService(config or VoiceConfig(), engine_factory, **kwargs)


def test_announces_and_speaks_with_fake_engine() -> None:
    engine = FakeEngine()
    service = build(lambda: engine, VoiceConfig(min_interval_seconds=0.0), start_worker=False)
    assert service.available is True
    assert service.mode == "tts"
    assert service.announce("One") is True
    assert service.announce("Two") is True
    service.flush()
    assert engine.say_calls == ["One", "Two"]
    assert engine.run_count == 2
    assert service.warning_count == 2
    service.stop()


def test_engine_properties_applied() -> None:
    engine = FakeEngine()
    service = build(lambda: engine, VoiceConfig(rate=160, volume=0.9), start_worker=False)
    assert engine.props.get("rate") == 160
    assert engine.props.get("volume") == 0.9
    service.stop()


def test_cooldown_blocks_repeated_warnings() -> None:
    engine = FakeEngine()
    service = build(
        lambda: engine,
        VoiceConfig(min_interval_seconds=100.0),
        start_worker=False,
    )
    assert service.announce("A") is True
    assert service.announce("B") is False
    assert service.warning_count == 1
    service.flush()
    assert engine.say_calls == ["A"]
    service.stop()


def test_reset_clears_cooldown() -> None:
    engine = FakeEngine()
    service = build(
        lambda: engine,
        VoiceConfig(min_interval_seconds=100.0),
        start_worker=False,
    )
    assert service.announce("A") is True
    assert service.announce("B") is False
    service.reset()
    assert service.announce("C") is True
    assert service.warning_count == 2
    service.stop()


def test_disabled_service_never_warns() -> None:
    service = build(
        lambda: FakeEngine(),
        VoiceConfig(enabled=False, min_interval_seconds=0.0),
        start_worker=False,
    )
    assert service.announce("A") is False
    assert service.warning_count == 0
    service.stop()


def test_engine_failure_falls_back_to_beep() -> None:
    def broken_factory():
        raise RuntimeError("no audio device")

    service = build(broken_factory, VoiceConfig(fallback="beep"), start_worker=False)
    assert service.mode == "beep"
    assert service.available is True
    assert service.announce("A") is True  # schedules a beep, never raises
    assert service.warning_count == 1
    service.flush()
    service.stop()


def test_stop_releases_engine_gracefully() -> None:
    engine = FakeEngine()
    service = build(lambda: engine, VoiceConfig(min_interval_seconds=0.0), start_worker=True)
    assert service.announce("A") is True
    service.stop()
    assert engine.stop_count >= 1