from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..runner import DEFAULT_ROM_PATH
from .actions import NUM_ACTIONS, action_from_index
from .adapter import EmulatorBackend, GalagaEmulatorAdapter
from .observations import OBSERVATION_DIM, encode_observation
from .rewards import GalagaRewardCalculator, RewardConfig


class GalagaRLEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for RAM/OAM-based RL training on NES Galaga."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str = DEFAULT_ROM_PATH,
        *,
        frames_per_decision: int = 4,
        max_steps: int = 4500,
        random_noop_max: int = 0,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        adapter: EmulatorBackend | None = None,
        capture_step_frames: bool = False,
        capture_audio_events: bool = False,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.rom_path = rom_path
        self.frames_per_decision = frames_per_decision
        self.max_steps = max_steps
        self.random_noop_max = random_noop_max
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        adapter_render_mode = render_mode if render_mode in {"rgb_array", "human"} else "rgb_array"
        self._adapter: EmulatorBackend = adapter or GalagaEmulatorAdapter(
            rom_path=rom_path,
            frames_per_decision=frames_per_decision,
            render_mode=adapter_render_mode,
            random_noop_max=random_noop_max,
            capture_step_frames=capture_step_frames,
            capture_audio_events=capture_audio_events,
        )
        self._reward_calc = GalagaRewardCalculator(reward_config)
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_snapshot = None

    @property
    def last_snapshot(self) -> Any:
        return self._last_snapshot

    @property
    def last_frame(self) -> Any:
        return self._adapter.last_frame

    @property
    def last_step_frames(self) -> tuple[Any, ...]:
        frames = getattr(self._adapter, "last_step_frames", None)
        if frames:
            return tuple(frames)
        frame = self._adapter.last_frame
        return (frame,) if frame is not None else ()

    @property
    def last_step_audio_events(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        events = getattr(self._adapter, "last_step_audio_events", None)
        return tuple(events) if events else ()

    @property
    def ram(self) -> Any:
        return getattr(self._adapter, "ram", None)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        snapshot, raw_info = self._adapter.reset(seed=seed)
        self._reward_calc.reset(snapshot)
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_snapshot = snapshot

        obs = encode_observation(snapshot)
        info = {
            "rom_path": self.rom_path,
            "x": snapshot.x,
            "score": snapshot.score,
            "lives": snapshot.lives,
            "stage": snapshot.stage,
            "clear": snapshot.clear,
            "dead": snapshot.dead,
            "steps": self._step_count,
            "episode_reward": self._episode_reward,
            "raw_info": raw_info,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_snapshot is None:
            raise RuntimeError("Environment must be reset() before step()")

        previous = self._last_snapshot
        step_result = self._adapter.step(int(action))
        current = step_result.snapshot
        self._step_count += 1

        breakdown = self._reward_calc.compute(
            previous,
            current,
            died=step_result.died,
            cleared=step_result.cleared,
        )
        self._episode_reward += breakdown.total
        self._last_snapshot = current

        terminated = bool(step_result.terminated or current.dead or current.clear)
        truncated = bool(
            step_result.truncated or (not terminated and self._step_count >= self.max_steps)
        )

        obs = encode_observation(current)
        info = {
            "rom_path": self.rom_path,
            "action": action_from_index(int(action)).value,
            "x": current.x,
            "score": current.score,
            "lives": current.lives,
            "stage": current.stage,
            "clear": current.clear,
            "dead": current.dead,
            "steps": self._step_count,
            "episode_reward": self._episode_reward,
            "reward_breakdown": breakdown.to_dict(),
            "native_reward": step_result.native_reward,
            "frames_executed": step_result.frames_executed,
            "raw_info": step_result.info,
        }
        return obs, float(breakdown.total), terminated, truncated, info

    def render(self) -> Any:
        return self._adapter.render()

    def close(self) -> None:
        self._adapter.close()
