from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..actions import Action
from ..audio import RomSoundTracker
from ..runner import DEFAULT_ROM_PATH, _unwrap_ram, create_galaga_env
from ..state import GalagaSnapshot, GalagaStateParser
from .actions import action_from_index, expand_macro_frames


@dataclass(frozen=True, slots=True)
class AdapterStepResult:
    snapshot: GalagaSnapshot
    native_reward: float
    terminated: bool
    truncated: bool
    died: bool
    cleared: bool
    info: dict[str, Any]
    frames_executed: int


class EmulatorBackend(Protocol):
    @property
    def last_snapshot(self) -> GalagaSnapshot | None: ...

    @property
    def last_frame(self) -> Any: ...

    def reset(self, *, seed: int | None = None) -> tuple[GalagaSnapshot, dict[str, Any]]: ...

    def step(self, action_index: int) -> AdapterStepResult: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class GalagaEmulatorAdapter:
    """Adapter bridging nes_py Galaga ROM, RAM/OAM parsing, and RL macro actions."""

    def __init__(
        self,
        rom_path: str = DEFAULT_ROM_PATH,
        *,
        frames_per_decision: int = 4,
        render_mode: str = "rgb_array",
        random_noop_max: int = 0,
        goal: str = "Survive and destroy all Galaga enemies for the highest score.",
        capture_step_frames: bool = False,
        capture_audio_events: bool = False,
    ) -> None:
        if frames_per_decision < 1:
            raise ValueError("frames_per_decision must be at least 1")
        if random_noop_max < 0:
            raise ValueError("random_noop_max must be non-negative")

        self.rom_path = rom_path
        self.frames_per_decision = frames_per_decision
        self.render_mode = render_mode
        self.random_noop_max = random_noop_max
        self.capture_step_frames = capture_step_frames
        self.capture_audio_events = capture_audio_events
        self.parser = GalagaStateParser(
            goal=goal,
            decision_horizon_frames=frames_per_decision,
        )
        self._env: Any | None = None
        self._rng = np.random.default_rng()
        self._last_frame: Any = None
        self._last_step_frames: list[Any] = []
        self._last_step_audio_events: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._sound_tracker = RomSoundTracker()
        self._last_snapshot: GalagaSnapshot | None = None
        self._previous_action: Action | None = None

    @property
    def last_snapshot(self) -> GalagaSnapshot | None:
        return self._last_snapshot

    @property
    def last_frame(self) -> Any:
        return self._last_frame

    @property
    def last_step_frames(self) -> tuple[Any, ...]:
        return tuple(self._last_step_frames)

    @property
    def last_step_audio_events(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return tuple(self._last_step_audio_events)

    @property
    def ram(self) -> Any:
        return _unwrap_ram(self._env) if self._env is not None else None

    def _ensure_env(self) -> Any:
        if self._env is None:
            self._env = create_galaga_env(self.rom_path, render_mode=self.render_mode)
        return self._env

    def reset(self, *, seed: int | None = None) -> tuple[GalagaSnapshot, dict[str, Any]]:
        env = self._ensure_env()
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        frame, info = env.reset(seed=seed)
        self.parser.reset()
        self._sound_tracker.reset()
        self._last_frame = frame
        self._last_step_frames = [frame] if self.capture_step_frames else []
        self._last_step_audio_events = []
        self._previous_action = None

        if self.random_noop_max > 0:
            noops = int(self._rng.integers(0, self.random_noop_max + 1))
            for _ in range(noops):
                frame, _, terminated, truncated, info = env.step(0)
                self._last_frame = frame
                if terminated or truncated:
                    break

        ram = _unwrap_ram(env)
        if self.capture_audio_events:
            self._sound_tracker.sync(ram)

        snapshot = self.parser.parse(info, ram)
        self._last_snapshot = snapshot
        return snapshot, dict(info)

    def step(self, action_index: int) -> AdapterStepResult:
        env = self._ensure_env()
        action = action_from_index(action_index)
        frame_actions = expand_macro_frames(
            action,
            frames=self.frames_per_decision,
            previous_action=self._previous_action,
        )

        native_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        frames_executed = 0
        if self.capture_step_frames:
            self._last_step_frames = []
        if self.capture_audio_events:
            self._last_step_audio_events = []

        ram = _unwrap_ram(env)
        for frame_act in frame_actions:
            frame, reward, terminated, truncated, info = env.step(frame_act)
            self._last_frame = frame
            if self.capture_step_frames:
                self._last_step_frames.append(frame)
            if self.capture_audio_events:
                self._last_step_audio_events.append(self._sound_tracker.sync(ram))
            native_reward += float(reward)
            frames_executed += 1
            if terminated or truncated:
                break

        snapshot = self.parser.parse(
            info,
            ram,
            previous_action=action.value,
            previous_reward=native_reward,
        )
        died = snapshot.dead
        cleared = snapshot.clear

        self._previous_action = action
        self._last_snapshot = snapshot

        return AdapterStepResult(
            snapshot=snapshot,
            native_reward=native_reward,
            terminated=bool(terminated or died or cleared),
            truncated=bool(truncated),
            died=died,
            cleared=cleared,
            info=info,
            frames_executed=frames_executed,
        )

    def render(self) -> Any:
        if self._env is None:
            return self._last_frame
        return self._env.render()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
