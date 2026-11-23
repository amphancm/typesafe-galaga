from __future__ import annotations

import numpy as np

from typesafe_galaga.rl.adapter import AdapterStepResult
from typesafe_galaga.rl.env import GalagaRLEnv
from typesafe_galaga.rl.observations import OBSERVATION_DIM, encode_observation
from typesafe_galaga.state import GalagaEnemy, GalagaProjectile, GalagaSnapshot


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
        "enemies": (
            GalagaEnemy(
                x=100, y=60, dx_from_player=0, dy_from_player=139, diving=False, aligned=True
            ),
        ),
        "projectiles": (GalagaProjectile(x=110, y=150, dx_from_player=10, dy_from_player=49),),
        "radar_grid": tuple("." * 11 for _ in range(9)),
    }
    defaults.update(kwargs)
    return GalagaSnapshot(**defaults)  # type: ignore[arg-type]


class _FakeAdapter:
    def __init__(self) -> None:
        self.last_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self.last_snapshot: GalagaSnapshot | None = None
        self._step = 0

    def reset(self, *, seed: int | None = None) -> tuple[GalagaSnapshot, dict[str, object]]:
        del seed
        self._step = 0
        self.last_snapshot = _snap(score=0)
        return self.last_snapshot, {"score": 0, "lives": 3}

    def step(self, action_index: int) -> AdapterStepResult:
        self._step += 1
        snap = _snap(score=self._step * 100, active_player_bullets=1 if action_index >= 3 else 0)
        self.last_snapshot = snap
        return AdapterStepResult(
            snapshot=snap,
            native_reward=100.0,
            terminated=False,
            truncated=False,
            died=False,
            cleared=False,
            info={"score": snap.score, "lives": snap.lives},
            frames_executed=4,
        )

    def render(self) -> np.ndarray:
        return self.last_frame

    def close(self) -> None:
        pass


def test_encode_observation_is_bounded_and_fixed_dim() -> None:
    obs = encode_observation(_snap())
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= -1.0
    assert float(obs.max()) <= 1.0


def test_encode_observation_populates_slots_and_radar() -> None:
    grid = ["." * 11 for _ in range(9)]
    grid[0] = "E" + "." * 10
    grid[8] = "." * 5 + "A" + "." * 5
    snap = _snap(
        previous_action="fire",
        player_bullets=((100, 150), (116, 100)),
        radar_grid=tuple(grid),
    )
    obs = encode_observation(snap)
    assert obs[15 + 3] == 1.0  # previous action one-hot for "fire"
    assert obs[19] == 1.0  # first enemy present
    assert obs[19 + 4 + 2] == 1.0  # second enemy empty-slot dy sentinel
    assert obs[19 + 24] == 1.0  # first projectile present
    assert obs[19 + 24 + 3 + 2] == 1.0  # second projectile empty-slot dy sentinel
    radar_start = 15 + 4 + 24 + 12
    assert obs[radar_start] == np.float32(0.5)  # "E" at (0, 0)
    assert obs[radar_start + 8 * 11 + 5] == np.float32(-1.0)  # "A" at (8, 5)


def test_galaga_rl_env_reset_and_step() -> None:
    env = GalagaRLEnv(adapter=_FakeAdapter(), max_steps=10)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBSERVATION_DIM,)
    assert env.action_space.n == 4
    assert info["lives"] == 3

    obs2, reward, terminated, truncated, info2 = env.step(3)
    assert obs2.shape == (OBSERVATION_DIM,)
    assert reward > 0.0
    assert terminated is False
    assert truncated is False
    assert info2["score"] == 100
    env.close()
