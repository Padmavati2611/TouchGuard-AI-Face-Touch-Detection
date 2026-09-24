"""Tests for the TouchGuardPipeline orchestrator."""

from __future__ import annotations

import math

import numpy as np

from touchguard.config import (
    CameraConfig,
    Settings,
    TouchConfig,
    VoiceConfig,
    default_settings,
)
from touchguard.datatypes import TouchStatus
from touchguard.pipeline import TouchGuardPipeline
from tests.conftest import (
    FakeVoice,
    StubDetector,
    face_box_at,
    idle_hand,
    released_hand,
    touching_hand,
)


def make_settings(persist: int = 3, clean: int = 15) -> Settings:
    settings = default_settings()
    settings.touch = TouchConfig(persist_frames=persist, clean_frames=clean)
    settings.voice = VoiceConfig(
        enabled=True, main_warning="Please avoid touching your face."
    )
    settings.camera = CameraConfig()
    return settings


def make_pipeline(**kwargs) -> TouchGuardPipeline:
    settings = kwargs.pop("settings", make_settings())
    return TouchGuardPipeline(
        settings,
        face_detector=kwargs.pop("face_detector", StubDetector(faces=[face_box_at()])),
        hand_detector=kwargs.pop("hand_detector", StubDetector(hands=[])),
        voice=kwargs.pop("voice", FakeVoice()),
        **kwargs,
    )


def test_process_happy_path_and_single_warning(blank_frame) -> None:
    voice = FakeVoice()
    face_detector = StubDetector(faces=[face_box_at()])
    hand_detector = StubDetector(hands=[])
    pipeline = make_pipeline(face_detector=face_detector, hand_detector=hand_detector, voice=voice)

    r1 = pipeline.process(blank_frame)
    assert len(r1.faces) == 1
    assert len(r1.hands) == 0
    assert r1.touching is False
    assert r1.status == TouchStatus.SAFE
    assert r1.should_warn is False
    assert r1.distance == math.inf

    hand_detector.hands = [touching_hand(face_box_at())]
    r2 = pipeline.process(blank_frame)
    assert r2.status == TouchStatus.FACE_TOUCH_DETECTED
    assert r2.should_warn is False
    assert r2.touch_count == 0

    r3 = pipeline.process(blank_frame)
    assert r3.status == TouchStatus.FACE_TOUCH_DETECTED

    r4 = pipeline.process(blank_frame)
    assert r4.status == TouchStatus.FACE_TOUCH_DETECTED
    assert r4.should_warn is True
    assert r4.touch_count == 1
    assert voice.warning_count == 1

    r5 = pipeline.process(blank_frame)
    assert r5.should_warn is False
    assert r5.touching is True
    assert r5.touch_count == 1
    assert voice.warning_count == 1
    assert voice.messages == ["Please avoid touching your face."]


def test_process_detects_approach_state(blank_frame) -> None:
    voice = FakeVoice()
    from tests.conftest import approaching_hand

    hand_detector = StubDetector(hands=[approaching_hand(face_box_at())])
    pipeline = make_pipeline(
        settings=make_settings(persist=1), hand_detector=hand_detector, voice=voice
    )
    analysis = pipeline.process(blank_frame)
    assert analysis.status == TouchStatus.HAND_APPROACHING
    assert analysis.should_warn is False
    assert voice.warning_count == 0


def test_process_returns_annotated_ndarray(blank_frame) -> None:
    pipeline = make_pipeline()
    analysis = pipeline.process(blank_frame)
    assert isinstance(analysis.frame, np.ndarray)
    assert analysis.frame.shape == blank_frame.shape
    assert analysis.fps > 0.0


def test_voice_disabled_never_warns(blank_frame) -> None:
    voice = FakeVoice(enabled=False)
    hand_detector = StubDetector(hands=[])
    pipeline = make_pipeline(
        settings=make_settings(persist=1),
        hand_detector=hand_detector,
        voice=voice,
    )
    hand_detector.hands = [touching_hand(face_box_at())]
    analysis = pipeline.process(blank_frame)
    assert analysis.should_warn is True
    assert analysis.status == TouchStatus.FACE_TOUCH_DETECTED
    assert analysis.voice_enabled is False
    assert voice.warning_count == 0


def test_no_face_status_propagates_through_pipeline(blank_frame) -> None:
    pipeline = make_pipeline(face_detector=StubDetector(faces=[]))
    analysis = pipeline.process(blank_frame)
    assert analysis.status == TouchStatus.NO_FACE_DETECTED
    assert analysis.touching is False


def test_close_releases_voice_and_hand_services() -> None:
    voice = FakeVoice()
    hand = StubDetector(hands=[])
    pipeline = make_pipeline(hand_detector=hand, voice=voice)
    assert voice.stop_called is False
    assert hand.close_called is False
    pipeline.close()
    assert voice.stop_called is True
    assert hand.close_called is True
    pipeline.close()  # idempotent


def test_model_status_reports_availability() -> None:
    pipeline = make_pipeline()
    status = pipeline.model_status
    assert set(status) == {"face", "hand", "voice"}
    assert "available" in status["face"]
    assert status["voice"]["mode"] == "fake"


def test_reset_clears_logic_and_cooldown(blank_frame) -> None:
    hand_detector = StubDetector(hands=[])
    pipeline = make_pipeline(settings=make_settings(persist=1), hand_detector=hand_detector)
    hand_detector.hands = [touching_hand(face_box_at())]
    analysis = pipeline.process(blank_frame)
    assert analysis.should_warn is True
    pipeline.reset()
    analysis = pipeline.process(blank_frame)
    assert analysis.should_warn is True  # re-armed after reset


def test_idle_hand_annotation_has_distance_line(blank_frame) -> None:
    hand_detector = StubDetector(hands=[idle_hand(face_box_at())])
    pipeline = make_pipeline(hand_detector=hand_detector)
    analysis = pipeline.process(blank_frame)
    assert analysis.closest_point is not None
    assert analysis.region_point is not None
    assert analysis.region_name in (
        "forehead",
        "nose",
        "left_cheek",
        "right_cheek",
        "mouth_chin",
    )


def test_repeat_warning_while_hand_keeps_touching(blank_frame) -> None:
    """The spoken warning repeats every repeat_interval_seconds to keep
    reminding (with the running total) until the hand is removed."""
    import time

    hand = StubDetector(hands=[])
    settings = make_settings(persist=1, clean=1)
    settings.voice = VoiceConfig(
        enabled=True,
        main_warning="Face touch {count}",
        repeat_interval_seconds=0.02,
    )
    voice = FakeVoice()
    pipeline = make_pipeline(settings=settings, hand_detector=hand, voice=voice)

    hand.hands = [touching_hand(face_box_at())]
    first = pipeline.process(blank_frame)  # new touch -> warning #1
    assert first.should_warn is True
    assert voice.messages == ["Face touch 1"]

    time.sleep(0.05)                      # still touching past the interval
    for _ in range(3):
        analysis = pipeline.process(blank_frame)
        assert analysis.touching is True
        assert analysis.should_warn is False
        assert analysis.touch_count == 1
    assert voice.warning_count == 2        # repeated reminder, same total
    assert voice.messages[-1] == "Face touch 1"

    hand.hands = [released_hand(face_box_at())]   # pull the hand away
    pipeline.process(blank_frame)
    hand.hands = [touching_hand(face_box_at())]   # touch again -> count 2
    analysis = pipeline.process(blank_frame)
    assert analysis.should_warn is True
    assert analysis.touch_count == 2
    assert voice.messages[-1] == "Face touch 2"


def test_repeat_warning_disabled_when_interval_zero(blank_frame) -> None:
    import time

    hand = StubDetector(hands=[])
    settings = make_settings(persist=1)
    settings.voice = VoiceConfig(
        enabled=True,
        main_warning="Face touch {count}",
        repeat_interval_seconds=0.0,       # 0 disables repeats
    )
    voice = FakeVoice()
    pipeline = make_pipeline(settings=settings, hand_detector=hand, voice=voice)

    hand.hands = [touching_hand(face_box_at())]
    assert pipeline.process(blank_frame).should_warn is True
    time.sleep(0.05)
    analysis = pipeline.process(blank_frame)
    assert analysis.touching is True
    assert voice.warning_count == 1        # still only the original warning


def test_default_message_speaks_touch_number(blank_frame) -> None:
    """The default voice message numbers each new touch event."""
    from touchguard.config import VoiceConfig

    voice = FakeVoice()
    hand = StubDetector(hands=[])
    settings = make_settings(persist=1, clean=1)
    settings.voice = VoiceConfig(
        enabled=True,
        repeat_interval_seconds=0.0,
        main_warning="Warning. You touched your face. Touch number {count}.",
    )
    pipeline = make_pipeline(settings=settings, hand_detector=hand, voice=voice)

    face = face_box_at()
    touching = touching_hand(face)
    released = released_hand(face)

    hand.hands = [touching]
    assert pipeline.process(blank_frame).touch_count == 1
    hand.hands = [released]
    pipeline.process(blank_frame)  # re-arm
    hand.hands = [touching]
    assert pipeline.process(blank_frame).touch_count == 2

    assert voice.messages == [
        "Warning. You touched your face. Touch number 1.",
        "Warning. You touched your face. Touch number 2.",
    ]


def test_every_separate_touch_warns_with_default_voice(blank_frame) -> None:
    """A continuous hold warns exactly once; each new touch warns again.

    This encodes the acceptance behaviour: Touch -> hold -> remove == 1
    warning, and Touch -> remove x3 == 3 warnings.
    """
    voice = FakeVoice()
    hand = StubDetector(hands=[])
    settings = make_settings(persist=1, clean=1)
    settings.voice = VoiceConfig(
        enabled=True,
        main_warning="Warning! Please keep your hands away from your face.",
        min_interval_seconds=0.0,          # never silently drop a new touch
        repeat_interval_seconds=0.0,       # never re-warn during a hold
    )
    pipeline = make_pipeline(settings=settings, hand_detector=hand, voice=voice)

    face = face_box_at()
    touching = touching_hand(face)
    released = released_hand(face)

    def touch():
        hand.hands = [touching]
        return pipeline.process(blank_frame)

    def release():
        hand.hands = [released]
        return pipeline.process(blank_frame)

    # Touch #1 -> exactly one warning while the hand stays on the face.
    assert touch().should_warn is True
    assert touch().should_warn is False    # still holding: no second warning
    assert touch().should_warn is False
    assert touch().touch_count == 1
    assert voice.warning_count == 1

    # Remove the hand -> system re-arms for a brand-new event.
    release()

    # Touch #2 -> a fresh warning with an updated total.
    second = touch()
    assert second.should_warn is True
    assert second.touch_count == 2
    assert voice.warning_count == 2

    release()

    # Touch #3 -> a fresh warning with an updated total.
    third = touch()
    assert third.should_warn is True
    assert third.touch_count == 3
    assert voice.warning_count == 3

    assert voice.messages == [
        "Warning! Please keep your hands away from your face."
    ] * 3