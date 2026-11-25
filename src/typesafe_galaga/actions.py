from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    NOOP = "noop"
    LEFT = "left"
    RIGHT = "right"
    FIRE = "fire"
    LEFT_FIRE = "left_fire"
    RIGHT_FIRE = "right_fire"


# Ordered 6-action space for NES Galaga via JoypadSpace.
ACTION_TO_INDEX: dict[Action, int] = {
    Action.NOOP: 0,
    Action.LEFT: 1,
    Action.RIGHT: 2,
    Action.FIRE: 3,
    Action.LEFT_FIRE: 4,
    Action.RIGHT_FIRE: 5,
}

JOYPAD_MOVEMENT: list[list[str]] = [
    ["NOOP"],
    ["left"],
    ["right"],
    ["A"],
    ["left", "A"],
    ["right", "A"],
]

FIRE_ACTIONS = frozenset({Action.FIRE, Action.LEFT_FIRE, Action.RIGHT_FIRE})

FIRE_RELEASE_ACTION: dict[Action, Action] = {
    Action.FIRE: Action.NOOP,
    Action.LEFT_FIRE: Action.LEFT,
    Action.RIGHT_FIRE: Action.RIGHT,
}


ACTION_DESCRIPTIONS: dict[Action, str] = {
    Action.NOOP: "Hold position and do not fire.",
    Action.LEFT: "Move the starfighter left without firing.",
    Action.RIGHT: "Move the starfighter right without firing.",
    Action.FIRE: "Hold position and fire a shot upward.",
    Action.LEFT_FIRE: "Move the starfighter left while firing.",
    Action.RIGHT_FIRE: "Move the starfighter right while firing.",
}
