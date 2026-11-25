from __future__ import annotations

from collections.abc import Sequence

from ..actions import ACTION_TO_INDEX, FIRE_ACTIONS, FIRE_RELEASE_ACTION, Action

# Canonical ordered 6-action space for NES Galaga.
RL_ACTIONS: tuple[Action, ...] = tuple(
    action for action, _ in sorted(ACTION_TO_INDEX.items(), key=lambda item: item[1])
)
NUM_ACTIONS: int = len(RL_ACTIONS)


def action_from_index(index: int) -> Action:
    """Map an integer RL action index (0..5) to a Galaga Action enum."""
    idx = int(index)
    if idx < 0 or idx >= NUM_ACTIONS:
        raise ValueError(f"Action index {idx} out of range [0, {NUM_ACTIONS - 1}]")
    return RL_ACTIONS[idx]


def index_from_action(action: Action | str) -> int:
    """Map an Action enum or its string value to an integer RL action index."""
    resolved = action if isinstance(action, Action) else Action(action)
    return ACTION_TO_INDEX[resolved]


def expand_macro_frames(
    action: Action,
    *,
    frames: int,
    previous_action: Action | None = None,
) -> list[int]:
    """Expand a macro action into per-frame JoypadSpace indices.

    On NES Galaga, holding the A button continuously only fires one shot on the
    rising edge. When a fire macro is selected after a previous fire macro, emit
    a single button-up release frame first so a fresh shot edge is triggered.
    """
    if frames < 1:
        raise ValueError("frames must be at least 1")

    action_index = ACTION_TO_INDEX[action]
    needs_release_edge = action in FIRE_ACTIONS and previous_action in FIRE_ACTIONS and frames > 1
    if not needs_release_edge:
        return [action_index] * frames

    release_index = ACTION_TO_INDEX[FIRE_RELEASE_ACTION[action]]
    return [release_index] + [action_index] * (frames - 1)


def action_names() -> Sequence[str]:
    """Return ordered human-readable names for the RL action space."""
    return tuple(action.value for action in RL_ACTIONS)
