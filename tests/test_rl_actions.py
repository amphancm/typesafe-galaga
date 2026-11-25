from __future__ import annotations

import pytest

from typesafe_galaga.actions import FIRE_ACTIONS, FIRE_RELEASE_ACTION, JOYPAD_MOVEMENT, Action
from typesafe_galaga.rl.actions import (
    NUM_ACTIONS,
    RL_ACTIONS,
    action_from_index,
    action_names,
    expand_macro_frames,
    index_from_action,
)


def test_galaga_action_space_matches_joypad_movement() -> None:
    assert NUM_ACTIONS == 6
    assert len(RL_ACTIONS) == 6
    assert len(JOYPAD_MOVEMENT) == 6
    assert action_names() == ("noop", "left", "right", "fire", "left_fire", "right_fire")
    for idx, action in enumerate(RL_ACTIONS):
        assert action_from_index(idx) is action
        assert index_from_action(action) == idx
        assert index_from_action(action.value) == idx
    assert FIRE_RELEASE_ACTION[Action.LEFT_FIRE] is Action.LEFT
    assert FIRE_RELEASE_ACTION[Action.RIGHT_FIRE] is Action.RIGHT


def test_action_from_index_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        action_from_index(-1)
    with pytest.raises(ValueError):
        action_from_index(NUM_ACTIONS)


def test_expand_macro_frames_inserts_release_edge_for_consecutive_fire() -> None:
    for fire_action in FIRE_ACTIONS:
        frames = expand_macro_frames(
            fire_action,
            frames=4,
            previous_action=Action.FIRE,
        )
        expected_release = index_from_action(FIRE_RELEASE_ACTION[fire_action])
        expected_hold = index_from_action(fire_action)
        assert frames == [expected_release, expected_hold, expected_hold, expected_hold]

    no_release = expand_macro_frames(
        Action.FIRE,
        frames=4,
        previous_action=Action.LEFT,
    )
    assert no_release == [index_from_action(Action.FIRE)] * 4
