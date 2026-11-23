from __future__ import annotations

from dataclasses import dataclass

from ..state import GalagaSnapshot


@dataclass(frozen=True, slots=True)
class RewardConfig:
    score_scale: float = 0.01
    score_bonus_cap: float = 10.0
    survival_bonus: float = 0.01
    alignment_bonus: float = 0.02
    fire_when_aligned_bonus: float = 0.05
    life_lost_penalty: float = -10.0
    death_penalty: float = -20.0
    stage_clear_bonus: float = 25.0


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    score: float
    survival: float
    alignment: float
    life_lost: float
    death: float
    stage_clear: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "score": self.score,
            "survival": self.survival,
            "alignment": self.alignment,
            "life_lost": self.life_lost,
            "death": self.death,
            "stage_clear": self.stage_clear,
        }


class GalagaRewardCalculator:
    """Shaped reward function for NES Galaga RL training."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def reset(self, initial_snapshot: GalagaSnapshot) -> None:
        del initial_snapshot

    def compute(
        self,
        previous: GalagaSnapshot,
        current: GalagaSnapshot,
        *,
        died: bool = False,
        cleared: bool = False,
    ) -> RewardBreakdown:
        cfg = self.config
        is_dead = bool(died or current.dead)
        is_clear = bool(cleared or current.clear)

        delta_score = max(0, current.score - previous.score)
        score_reward = min(delta_score * cfg.score_scale, cfg.score_bonus_cap)

        survival_reward = cfg.survival_bonus if current.alive and not is_dead else 0.0

        alignment_reward = 0.0
        if current.aligned_enemy_count > 0 and not current.danger_imminent:
            alignment_reward += cfg.alignment_bonus
            if current.active_player_bullets > previous.active_player_bullets:
                alignment_reward += cfg.fire_when_aligned_bonus

        life_lost_reward = cfg.life_lost_penalty if current.lives < previous.lives else 0.0
        death_reward = cfg.death_penalty if is_dead else 0.0
        stage_clear_reward = cfg.stage_clear_bonus if is_clear else 0.0

        total = (
            score_reward
            + survival_reward
            + alignment_reward
            + life_lost_reward
            + death_reward
            + stage_clear_reward
        )
        return RewardBreakdown(
            total=float(total),
            score=float(score_reward),
            survival=float(survival_reward),
            alignment=float(alignment_reward),
            life_lost=float(life_lost_reward),
            death=float(death_reward),
            stage_clear=float(stage_clear_reward),
        )
