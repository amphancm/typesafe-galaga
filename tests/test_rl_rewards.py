from __future__ import annotations

from typesafe_galaga.rl.rewards import GalagaRewardCalculator, RewardConfig
from typesafe_galaga.state import GalagaSnapshot


def _snap(**kwargs: object) -> GalagaSnapshot:
    defaults: dict[str, object] = {
        "goal": "Survive",
        "stage": 1,
        "score": 0,
        "lives": 3,
        "x": 100,
        "y": 199,
        "dx": 0,
        "alive": True,
        "dead": False,
        "clear": False,
        "active_player_bullets": 0,
    }
    defaults.update(kwargs)
    return GalagaSnapshot(**defaults)  # type: ignore[arg-type]


def test_reward_calculator_rewards_score_and_penalizes_death() -> None:
    calc = GalagaRewardCalculator(RewardConfig())
    prev = _snap(score=100, lives=3)
    calc.reset(prev)

    scored = _snap(score=260, lives=3, aligned_enemy_count=1, active_player_bullets=1)
    b1 = calc.compute(prev, scored)
    assert b1.score > 1.0
    assert b1.survival > 0.0
    assert b1.alignment > 0.0
    assert b1.total > 1.0

    died = _snap(score=260, lives=2, alive=False, dead=True)
    b2 = calc.compute(scored, died, died=True)
    assert b2.life_lost < 0.0
    assert b2.death < 0.0
    assert b2.total < -20.0
