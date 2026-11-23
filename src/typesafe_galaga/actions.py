from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    NOOP = "noop"
    LEFT = "left"
    RIGHT = "right"
    FIRE = "fire"


# Ordered 4-action space for NES Galaga via JoypadSpace.
ACTION_TO_INDEX: dict[Action, int] = {
    Action.NOOP: 0,
    Action.LEFT: 1,
    Action.RIGHT: 2,
    Action.FIRE: 3,
}

JOYPAD_MOVEMENT: list[list[str]] = [
    ["NOOP"],
    ["left"],
    ["right"],
    ["A"],
]

FIRE_ACTIONS = frozenset({Action.FIRE})

FIRE_RELEASE_ACTION: dict[Action, Action] = {
    Action.FIRE: Action.NOOP,
}


ACTION_DESCRIPTIONS: dict[Action, str] = {
    Action.NOOP: "Hold position and do not fire.",
    Action.LEFT: "Move the starfighter left without firing.",
    Action.RIGHT: "Move the starfighter right without firing.",
    Action.FIRE: "Hold position and fire a shot upward.",
}
