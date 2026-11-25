from __future__ import annotations

import numpy as np
import pytest

from typesafe_galaga.audio import AUDIO_EVENT_NAMES, ROM_SOUND_STATUS_ADDR, extract_ram_audio_events
from typesafe_galaga.rl.adapter import AdapterStepResult
from typesafe_galaga.rl.env import GalagaRLEnv
from typesafe_galaga.rl.observations import (
    AUDIO_EVENT_DIM,
    AUDIO_EVENT_INDEX,
    BASE_OBSERVATION_DIM,
    OBSERVATION_DIM,
    FrameStackBuffer,
    encode_observation,
    infer_include_audio_from_shape,
    infer_obs_config_from_shape,
)
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
    enemy_start = 15 + 6
    assert obs[enemy_start] == 1.0  # first enemy present
    assert obs[enemy_start + 4 + 2] == 1.0  # second enemy empty-slot dy sentinel
    assert obs[enemy_start + 24] == 1.0  # first projectile present
    assert obs[enemy_start + 24 + 3 + 2] == 1.0  # second projectile empty-slot dy sentinel
    radar_start = enemy_start + 24 + 12
    assert obs[radar_start] == np.float32(0.5)  # "E" at (0, 0)
    assert obs[radar_start + 8 * 11 + 5] == np.float32(-1.0)  # "A" at (8, 5)


def test_galaga_rl_env_reset_and_step() -> None:
    env = GalagaRLEnv(adapter=_FakeAdapter(), max_steps=10)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBSERVATION_DIM,)
    assert env.action_space.n == 6
    assert info["lives"] == 3

    obs2, reward, terminated, truncated, info2 = env.step(3)
    assert obs2.shape == (OBSERVATION_DIM,)
    assert reward > 0.0
    assert terminated is False
    assert truncated is False
    assert info2["score"] == 100
    # Fallback audio detection from snapshot transition (score increased + bullet fired)
    assert "laser_shot" in info2["audio_events"]
    assert "enemy_hit" in info2["audio_events"]
    assert obs2[BASE_OBSERVATION_DIM + AUDIO_EVENT_INDEX["laser_shot"]] == 1.0
    assert obs2[BASE_OBSERVATION_DIM + AUDIO_EVENT_INDEX["enemy_hit"]] == 1.0
    env.close()


def test_encode_observation_audio_flags_and_legacy_mode() -> None:
    assert OBSERVATION_DIM == BASE_OBSERVATION_DIM + AUDIO_EVENT_DIM
    assert AUDIO_EVENT_DIM == len(AUDIO_EVENT_NAMES) == 10

    snap = _snap(audio_events=("laser_shot", "dive_swoop", "ship_explode"))
    obs_with_audio = encode_observation(snap, include_audio=True)
    assert obs_with_audio.shape == (OBSERVATION_DIM,)
    audio_slice = obs_with_audio[BASE_OBSERVATION_DIM:]
    assert audio_slice[AUDIO_EVENT_INDEX["laser_shot"]] == 1.0
    assert audio_slice[AUDIO_EVENT_INDEX["dive_swoop"]] == 1.0
    assert audio_slice[AUDIO_EVENT_INDEX["ship_explode"]] == 1.0
    assert audio_slice[AUDIO_EVENT_INDEX["enemy_hit"]] == 0.0
    assert float(audio_slice.sum()) == 3.0

    obs_legacy = encode_observation(snap, include_audio=False)
    assert obs_legacy.shape == (BASE_OBSERVATION_DIM,)
    assert np.array_equal(obs_legacy, obs_with_audio[:BASE_OBSERVATION_DIM])

    assert infer_include_audio_from_shape((OBSERVATION_DIM,)) is True
    assert infer_include_audio_from_shape((BASE_OBSERVATION_DIM,)) is False
    assert infer_obs_config_from_shape((OBSERVATION_DIM * 4,)) == (True, 4)
    assert infer_obs_config_from_shape((BASE_OBSERVATION_DIM * 4,)) == (False, 4)
    with pytest.raises(ValueError, match="Unsupported observation dimension"):
        infer_include_audio_from_shape((99,))


def test_frame_stack_buffer_matches_sb3_layout() -> None:
    buf = FrameStackBuffer(obs_dim=3, n_stack=3)
    s0 = buf.reset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(s0, np.array([0, 0, 0, 0, 0, 0, 1, 2, 3], dtype=np.float32))
    s1 = buf.push(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert np.array_equal(s1, np.array([0, 0, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32))


def test_extract_ram_audio_events_combines_active_and_triggered_slots() -> None:
    ram = bytearray(0x0800)
    ram[ROM_SOUND_STATUS_ADDR + 32] = 2  # dive_swoop currently active
    ram[ROM_SOUND_STATUS_ADDR + 36] = 2  # ambient currently active
    events = extract_ram_audio_events(ram, started_slots=(34, 30))
    assert events == ("laser_shot", "enemy_hit", "dive_swoop", "ambient")


def test_galaga_rl_env_legacy_no_audio_obs_mode() -> None:
    env = GalagaRLEnv(adapter=_FakeAdapter(), max_steps=10, include_audio_obs=False)
    assert env.observation_space.shape == (BASE_OBSERVATION_DIM,)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (BASE_OBSERVATION_DIM,)
    obs2, _, _, _, _ = env.step(3)
    assert obs2.shape == (BASE_OBSERVATION_DIM,)
    env.close()
