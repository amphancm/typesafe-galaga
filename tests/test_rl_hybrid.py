from __future__ import annotations

from typesafe_galaga.actions import Action
from typesafe_galaga.policy import Decision
from typesafe_galaga.rl.hybrid import HybridConfig, HybridPolicy
from typesafe_galaga.state import GalagaSnapshot


class _StubPolicy:
    def __init__(self, action: Action, confidence: float) -> None:
        self.action = action
        self.confidence = confidence
        self.calls = 0

    def choose(self, snapshot: GalagaSnapshot, actions: tuple[Action, ...]) -> Decision:
        del snapshot, actions
        self.calls += 1
        return Decision(
            action=self.action,
            confidence=self.confidence,
            probabilities={self.action.value: self.confidence},
            latency_ms=1.0,
        )


def _snap(*, danger_imminent: bool = False) -> GalagaSnapshot:
    return GalagaSnapshot(
        goal="Survive",
        stage=1,
        score=0,
        lives=3,
        x=100,
        y=199,
        dx=0,
        alive=True,
        dead=False,
        clear=False,
        active_player_bullets=0,
        danger_imminent=danger_imminent,
    )


def test_hybrid_policy_escalates_on_low_confidence_or_danger() -> None:
    rl = _StubPolicy(Action.FIRE, confidence=0.40)
    jev = _StubPolicy(Action.LEFT, confidence=0.95)
    hybrid = HybridPolicy(
        rl,
        jev,
        config=HybridConfig(confidence_threshold=0.55, min_steps_between_jev_calls=2),
    )

    d1 = hybrid.choose(_snap(), tuple(Action))
    assert d1.action == Action.LEFT
    assert jev.calls == 1

    # Cooldown prevents immediate second escalation
    d2 = hybrid.choose(_snap(), tuple(Action))
    assert d2.action == Action.FIRE
    assert jev.calls == 1

    stats = hybrid.stats()
    assert stats.total_decisions == 2
    assert stats.jev_decisions == 1
    assert stats.rl_decisions == 1
