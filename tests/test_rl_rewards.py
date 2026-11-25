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


def test_reward_calculator_handles_large_score_and_stage_clear_progression() -> None:
    cfg = RewardConfig()
    calc = GalagaRewardCalculator(cfg)
    s0 = _snap(stage=1, score=0)
    calc.reset(s0)

    # Large bonus (e.g. Challenging Stage 5,000 pts) should not be clipped at 10.0.
    s1 = _snap(stage=1, score=5000)
    b_big = calc.compute(s0, s1)
    assert b_big.score == 50.0

    # Stage increment triggers stage_clear_bonus once.
    s2 = _snap(stage=2, score=5000)
    b_stage = calc.compute(s1, s2)
    assert b_stage.stage_clear == cfg.stage_clear_bonus

    # Audio stage_clear immediately after stage increment is suppressed by cooldown.
    s3 = _snap(stage=2, score=5000, audio_events=("stage_clear",))
    b_cooldown = calc.compute(s2, s3)
    assert b_cooldown.stage_clear == 0.0

    # Fresh calculator triggers stage_clear_bonus on rising edge of stage_clear audio event.
    calc2 = GalagaRewardCalculator(cfg)
    calc2.reset(s0)
    s_audio = _snap(stage=1, score=0, audio_events=("stage_clear",))
    b_audio = calc2.compute(s0, s_audio)
    assert b_audio.stage_clear == cfg.stage_clear_bonus
    b_audio_repeat = calc2.compute(s_audio, s_audio)
    assert b_audio_repeat.stage_clear == 0.0
