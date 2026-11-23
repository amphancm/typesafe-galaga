from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import ACTION_DESCRIPTIONS, Action
from .state import GalagaSnapshot


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    fire_needed_probability: float | None = None
    danger_score: float | None = None


class Policy(Protocol):
    def choose(self, snapshot: GalagaSnapshot, actions: Sequence[Action]) -> Decision: ...


class TypeSafePolicy:
    def __init__(self) -> None:
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        except ImportError as exc:
            raise RuntimeError(
                "typesafe-sdk is not installed. Install the project before using Jev."
            ) from exc
        self._Choice = Choice
        self._Noul = Noul
        self._Score = Score
        self._client = TypeSafeClient()
        self._next_action_instructions = {
            "question": "Which controller macro should the Galaga starfighter commit to next?",
            "goal": "Destroy enemies while dodging diving aliens and enemy projectiles.",
            "timing": "The selected action is held for the next decision horizon.",
            "evasion": (
                "Check `tactical.danger_imminent` and `hazards`. If `danger_imminent` is "
                "true, prioritize `tactical.recommended_evade_dir` (`left` or `right`) "
                "to dodge incoming projectiles or diving enemies."
            ),
            "offense": (
                "When safe, align `player.x` with `tactical.best_target_x` (check "
                "`tactical.best_target_dx`) using `left` or `right`, and use `fire` "
                "whenever aligned and `player.can_fire_more` is true."
            ),
        }
        self._fire_needed_question = Noul(
            instructions=(
                "Should the starfighter fire now? Yes when `player.can_fire_more` is true "
                "and enemies are present on screen, especially when aligned or under attack."
            )
        )
        self._danger_question = Score(
            instructions="How dangerous is the starfighter's immediate situation?",
            criteria=[
                "Safe open space",
                "Diving enemy or projectile approaching",
                "Immediate collision risk requiring evasion",
            ],
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _answer(response: Any, question_id: str, typed_collection: str) -> Any:
        collection = getattr(response, typed_collection, None)
        if collection is not None and question_id in collection:
            return collection[question_id]
        answers = getattr(response, "answers", None)
        if answers is not None and question_id in answers:
            return answers[question_id]
        raise KeyError(f"TypeSafe response omitted {question_id!r}")

    def choose(self, snapshot: GalagaSnapshot, actions: Sequence[Action]) -> Decision:
        criteria = {action.value: ACTION_DESCRIPTIONS[action] for action in actions}
        questions = {
            "next_action": self._Choice(
                instructions=self._next_action_instructions,
                criteria=criteria,
            ),
            "fire_needed": self._fire_needed_question,
            "danger": self._danger_question,
        }
        started = time.perf_counter()
        response = self._client.system_one(state=snapshot.to_state(), questions=questions)
        latency_ms = (time.perf_counter() - started) * 1000

        action_answer = self._answer(response, "next_action", "choices")
        fire_answer = self._answer(response, "fire_needed", "nouls")
        danger_answer = self._answer(response, "danger", "scores")
        action = Action(str(action_answer.choice))
        probabilities = {
            str(key): float(value) for key, value in dict(action_answer.probabilities).items()
        }
        return Decision(
            action=action,
            confidence=float(action_answer.confidence),
            probabilities=probabilities,
            latency_ms=latency_ms,
            fire_needed_probability=float(fire_answer.noul),
            danger_score=float(danger_answer.score),
        )


class HeuristicPolicy:
    """Offline Galaga baseline policy: evade imminent threats, align with targets, and fire."""

    def choose(self, snapshot: GalagaSnapshot, actions: Sequence[Action]) -> Decision:
        allowed = set(actions)
        can_fire = snapshot.active_player_bullets < 2

        if snapshot.danger_imminent:
            if snapshot.recommended_evade_dir == "left":
                desired = Action.LEFT
            elif snapshot.recommended_evade_dir == "right":
                desired = Action.RIGHT
            else:
                desired = Action.FIRE if can_fire else Action.NOOP
        elif snapshot.best_target_x is not None:
            dx = snapshot.best_target_x - snapshot.x
            if dx < -6:
                desired = Action.LEFT
            elif dx > 6:
                desired = Action.RIGHT
            else:
                desired = Action.FIRE if can_fire else Action.NOOP
        else:
            desired = Action.FIRE if can_fire else Action.NOOP

        action = desired if desired in allowed else actions[0]
        return Decision(
            action=action,
            confidence=1.0,
            probabilities={candidate.value: float(candidate == action) for candidate in actions},
            latency_ms=0.0,
            fire_needed_probability=1.0 if can_fire else 0.0,
            danger_score=2.0 if snapshot.danger_imminent else 0.0,
        )
