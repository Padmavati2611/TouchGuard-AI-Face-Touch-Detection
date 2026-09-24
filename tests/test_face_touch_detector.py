"""Tests for the distance-based face-touch detection state machine."""

from __future__ import annotations

import math

from touchguard.config import TouchConfig
from touchguard.datatypes import TouchStatus
from touchguard.face_touch_detector import FaceTouchLogic
from tests.conftest import (
    approaching_hand,
    face_box_at,
    hand_with_points,
    hovering_hand,
    idle_hand,
    released_hand,
    touching_hand,
)


def test_no_hands_means_safe() -> None:
    logic = FaceTouchLogic()
    decision = logic.update([face_box_at()], [])
    assert decision.touching is False
    assert decision.should_warn is False
    assert decision.status == TouchStatus.SAFE
    assert decision.distance == math.inf


def test_no_face_means_no_face_detected() -> None:
    logic = FaceTouchLogic()
    decision = logic.update([], [touching_hand(face_box_at())])
    assert decision.status == TouchStatus.NO_FACE_DETECTED
    assert decision.touching is False
    assert decision.should_warn is False


def test_no_face_never_creates_a_new_touch_event() -> None:
    """A vanishing face must not increment the count (requirement #9)."""
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    face = face_box_at()
    hand = touching_hand(face)

    assert logic.update([face], [hand]).should_warn is True
    assert logic.touch_count == 1

    for _ in range(5):
        decision = logic.update([], [hand])
        assert decision.status == TouchStatus.NO_FACE_DETECTED
        assert decision.should_warn is False
        assert decision.touch_count == 1


def test_no_face_rearms_then_next_touch_is_a_new_event() -> None:
    """After the face vanishes long enough the system re-arms; the next
    confirmed touch is a brand-new event with the updated count."""
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=2))
    face = face_box_at()
    hand = touching_hand(face)

    assert logic.update([face], [hand]).should_warn is True
    for _ in range(3):
        logic.update([], [hand])

    decision = logic.update([face], [hand])
    assert decision.should_warn is True
    assert decision.touch_count == 2


def test_no_hand_after_touch_rearms_for_next_event() -> None:
    """Requirement #10: a disappearing hand safely releases/re-arms the touch
    state so the next hand-touch counts as a fresh event."""
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=3))
    face = face_box_at()
    hand = touching_hand(face)

    assert logic.update([face], [hand]).should_warn is True
    assert logic.touch_count == 1

    # One no-hand frame must not immediately re-arm.
    assert logic.update([face], []).touch_active is True

    # Two more no-hand frames reach the clean period and re-arm.
    for _ in range(2):
        decision = logic.update([face], [])
        assert decision.status == TouchStatus.SAFE
        assert decision.should_warn is False
    assert logic.touch_active is False

    next_frame = logic.update([face], [hand])
    assert next_frame.should_warn is True
    assert next_frame.touch_count == 2


def test_fingertip_inside_face_counts_as_touch() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    face = face_box_at()
    decision = logic.update([face], [touching_hand(face)])
    assert decision.touching is True
    assert decision.should_warn is True
    assert decision.status == TouchStatus.FACE_TOUCH_DETECTED
    assert decision.distance == 0.0
    assert decision.touch_count == 1


def test_idle_hand_is_safe() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    decision = logic.update([face_box_at()], [idle_hand(face_box_at())])
    assert decision.touching is False
    assert decision.status == TouchStatus.SAFE


def test_approaching_hand_is_between_thresholds() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    face = face_box_at()
    decision = logic.update([face], [approaching_hand(face)])
    assert decision.status == TouchStatus.HAND_APPROACHING
    assert decision.touching is False
    assert decision.should_warn is False
    assert 90.0 < decision.distance <= 170.0


def test_persistence_fires_single_warning() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=3, clean_frames=15))
    face = face_box_at()
    hand = touching_hand(face)

    r1 = logic.update([face], [hand])
    assert r1.status == TouchStatus.FACE_TOUCH_DETECTED
    assert r1.should_warn is False

    r2 = logic.update([face], [hand])
    assert r2.status == TouchStatus.FACE_TOUCH_DETECTED
    assert r2.should_warn is False

    r3 = logic.update([face], [hand])
    assert r3.status == TouchStatus.FACE_TOUCH_DETECTED
    assert r3.should_warn is True
    assert r3.touch_active is True
    assert r3.touch_count == 1

    r4 = logic.update([face], [hand])  # still touching: no repeated warning
    assert r4.status == TouchStatus.FACE_TOUCH_DETECTED
    assert r4.should_warn is False
    assert r4.touch_count == 1


def test_hovering_too_close_does_not_rearm() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=3))
    face = face_box_at()

    assert logic.update([face], [touching_hand(face)]).should_warn is True

    # Hand hovers just outside the touch threshold but inside release margin.
    hover = logic.update([face], [hovering_hand(face)])
    assert hover.status == TouchStatus.HAND_APPROACHING
    assert hover.touch_active is True
    assert hover.should_warn is False

    # Returning to the face while touch is still active does NOT re-warn.
    again = logic.update([face], [touching_hand(face)])
    assert again.should_warn is False
    assert again.touch_count == 1


def test_rearms_only_after_clean_frames_and_warns_again() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=3))
    face = face_box_at()

    assert logic.update([face], [touching_hand(face)]).should_warn is True
    assert logic.touch_count == 1

    # Two frames outside the release margin are not enough to re-arm.
    assert logic.update([face], [released_hand(face)]).touch_active is True
    assert logic.update([face], [released_hand(face)]).touch_active is True

    # The third clean frame resets the event state.
    assert logic.update([face], [released_hand(face)]).touch_active is False

    # A brand-new touch produces a brand-new warning.
    decision = logic.update([face], [touching_hand(face)])
    assert decision.should_warn is True
    assert decision.touch_count == 2


def test_wrist_excluded_by_default() -> None:
    face = face_box_at()
    wrist_inside = hand_with_points(
        [(0, face.center)] + [(i, (600.0, 60.0)) for i in (4, 8, 12, 16, 20)]
    )
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    decision = logic.update([face], [wrist_inside])
    assert decision.touching is False
    assert decision.status == TouchStatus.SAFE


def test_wrist_included_when_configured() -> None:
    face = face_box_at()
    wrist_inside = hand_with_points(
        [(0, face.center)] + [(i, (600.0, 60.0)) for i in (4, 8, 12, 16, 20)]
    )
    logic = FaceTouchLogic(
        TouchConfig(persist_frames=1, clean_frames=1, use_wrist=True)
    )
    decision = logic.update([face], [wrist_inside])
    assert decision.touching is True
    assert decision.should_warn is True
    assert decision.status == TouchStatus.FACE_TOUCH_DETECTED


def test_multiple_hands_and_faces() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    face = face_box_at()
    decision = logic.update(
        [face, face_box_at(50, 50, 80, 80)], [idle_hand(face), touching_hand(face)]
    )
    assert decision.touching is True
    assert decision.status == TouchStatus.FACE_TOUCH_DETECTED
    assert decision.should_warn is True


def test_reset_clears_event_state() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=3, clean_frames=15))
    face = face_box_at()
    hand = touching_hand(face)
    for _ in range(3):
        last = logic.update([face], [hand])
    assert logic.touch_active is True
    assert logic.touch_count == 1
    assert last.should_warn is True
    logic.reset()
    assert logic.touch_active is False
    assert logic.touch_count == 0
    assert logic.touch_streak == 0
    last = None
    for _ in range(3):
        last = logic.update([face], [hand])
    assert logic.touch_active is True
    assert logic.touch_count == 1
    assert last.should_warn is True


def test_decision_tracks_streaks() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=5, clean_frames=10))
    face = face_box_at()
    hand = touching_hand(face)
    decision = logic.update([face], [hand])
    assert decision.touch_streak == 1
    assert decision.clean_streak == 0
    assert decision.touching is True


def test_distance_reports_and_regions_selected() -> None:
    logic = FaceTouchLogic(TouchConfig(persist_frames=1, clean_frames=1))
    face = face_box_at()
    decision = logic.update([face], [idle_hand(face)])
    assert decision.distance > 170.0
    assert decision.closest_point is not None
    assert decision.region_point is not None
    assert decision.closest_region_name in (
        "forehead",
        "nose",
        "left_cheek",
        "right_cheek",
        "mouth_chin",
    )