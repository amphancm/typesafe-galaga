from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, FIRE_ACTIONS, FIRE_RELEASE_ACTION, JOYPAD_MOVEMENT, Action
from .audio import AudioConfig, GalagaAudioEngine
from .dashboard import DashboardCommand, LiveDashboard
from .policy import Decision, Policy
from .state import OAM_END, OAM_START, PLAYER_Y, TILE_PLAYER_SHIP, GalagaStateParser, _read_score

DEFAULT_ROM_PATH = "roms/Galaga (U).nes"


def _unwrap_ram(env: Any) -> Any:
    ram = getattr(env, "ram", None)
    if ram is not None:
        return ram
    current = getattr(env, "env", None)
    visited: set[int] = {id(env)}
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        ram = getattr(current, "ram", None)
        if ram is not None:
            return ram
        current = getattr(current, "env", None)
    return None


def _ship_present(ram: Any) -> bool:
    if ram is None or len(ram) < OAM_END:
        return False
    try:
        oam = memoryview(ram)[OAM_START:OAM_END]
    except TypeError:
        for offset in range(OAM_START, OAM_END, 4):
            if int(ram[offset]) == PLAYER_Y and int(ram[offset + 1]) == TILE_PLAYER_SHIP:
                return True
        return False
    for offset in range(0, len(oam), 4):
        if oam[offset] == PLAYER_Y and oam[offset + 1] == TILE_PLAYER_SHIP:
            return True
    return False


class GalagaEnvWrapper:
    """Wraps nes_py.NESEnv + JoypadSpace with Galaga auto-start and life/score tracking."""

    def __init__(self, rom_path: str = DEFAULT_ROM_PATH, render_mode: str = "rgb_array") -> None:
        try:
            from nes_py import NESEnv
            from nes_py.wrappers import JoypadSpace
        except ImportError as exc:
            raise RuntimeError(
                'Galaga dependencies are missing. Install with: pip install -e ".[galaga]"'
            ) from exc

        path = Path(rom_path)
        if not path.exists():
            raise FileNotFoundError(f"Galaga ROM not found at: {path}")

        self._nes_env = NESEnv(str(path))
        self._env = JoypadSpace(self._nes_env, JOYPAD_MOVEMENT)
        self.render_mode = render_mode
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._lives = 3
        self._absent_frames = 0
        self._was_alive = True
        self._last_score = 0
        self._stage = 1
        self._has_start_backup = False

    @property
    def ram(self) -> Any:
        return self._nes_env.ram

    def _enrich_info(self, info: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        ram = self._nes_env.ram
        score = _read_score(ram)
        alive = _ship_present(ram)
        if alive:
            self._absent_frames = 0
            self._was_alive = True
        else:
            self._absent_frames += 1
            if self._was_alive and self._absent_frames >= 20:
                self._lives = max(0, self._lives - 1)
                self._was_alive = False

        dead = self._lives <= 0
        enriched = {
            **info,
            "score": score,
            "lives": self._lives,
            "stage": self._stage,
            "death": dead,
            "clear": False,
        }
        return enriched, dead

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del options
        frame, info = self._env.reset(seed=seed)
        self._lives = 3
        self._absent_frames = 0
        self._was_alive = True
        self._last_score = 0
        self._stage = 1

        if not self._has_start_backup:
            # Skip title screen and start 1-player game using Start button (bit 8).
            nes = self._nes_env
            for _ in range(110):
                nes._frame_advance(0)
            for _ in range(3):
                nes._frame_advance(8)
            for _ in range(15):
                nes._frame_advance(0)
            for _ in range(3):
                nes._frame_advance(8)

            ram = nes.ram
            for _ in range(750):
                nes._frame_advance(0)
                if _ship_present(ram):
                    break

            frame = nes.screen
            info = nes._get_info()
            nes._backup()
            self._has_start_backup = True

        enriched, _ = self._enrich_info(info)
        return frame, enriched

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        frame, _, terminated, truncated, info = self._env.step(int(action))
        if self.render_mode == "human":
            self._nes_env.render()
        enriched, dead = self._enrich_info(info)
        score = int(enriched["score"])
        reward = float(score - self._last_score)
        self._last_score = score
        if dead:
            terminated = True
        return frame, reward, bool(terminated), bool(truncated), enriched

    def render(self) -> Any:
        return self._nes_env.render()

    def close(self) -> None:
        self._env.close()


def create_galaga_env(
    rom_path: str = DEFAULT_ROM_PATH, render_mode: str = "human"
) -> GalagaEnvWrapper:
    return GalagaEnvWrapper(rom_path=rom_path, render_mode=render_mode)


def _record_decision(
    log: Any,
    *,
    decision_index: int,
    snapshot: Any,
    decision: Decision,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> None:
    record = {
        "decision": decision_index,
        "state": snapshot.to_state(),
        "debug_state": snapshot.to_debug_state(),
        "state_text": snapshot.to_text(),
        "action": decision.action.value,
        "confidence": decision.confidence,
        "probabilities": dict(decision.probabilities),
        "fire_needed_probability": decision.fire_needed_probability,
        "danger_score": decision.danger_score,
        "latency_ms": decision.latency_ms,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    log.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.flush()


def _run_realtime_dashboard(
    *,
    env: Any,
    dashboard: LiveDashboard,
    policy: Policy,
    parser: GalagaStateParser,
    frame: Any,
    info: dict[str, Any],
    log: Any,
    frames_per_decision: int,
    max_decisions: int,
    screenshot_path: Path | None,
    audio: GalagaAudioEngine | None = None,
) -> None:
    actions = tuple(Action)
    active_decision: Decision | None = None
    pending: Future[Decision] | None = None
    pending_snapshot: Any = None
    pending_index = 0
    pending_request_frame = 0
    next_index = 0
    frame_index = 0
    last_request_frame = -frames_per_decision
    episode_reward = 0.0
    reward_since_decision = 0.0
    previous_action: Action | None = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    previous_response_delay_frames = 0
    screenshot_saved = False
    terminated = truncated = False

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="typesafe-jev") as executor:
        while True:
            decision_updated = False
            snapshot = parser.parse(
                info,
                _unwrap_ram(env),
                previous_action=previous_action.value if previous_action else None,
                previous_reward=previous_reward,
                response_latency_ms=previous_latency_ms,
                response_delay_frames=previous_response_delay_frames,
            )

            if pending is not None and pending.done():
                active_decision = pending.result()
                _record_decision(
                    log,
                    decision_index=pending_index,
                    snapshot=pending_snapshot,
                    decision=active_decision,
                    reward=reward_since_decision,
                    terminated=terminated,
                    truncated=truncated,
                )
                print(
                    f"#{pending_index:04d} x={pending_snapshot.x:03d} "
                    f"score={pending_snapshot.score:06d} "
                    f"action={active_decision.action.value:<12} "
                    f"confidence={active_decision.confidence:.2f} "
                    f"latency={active_decision.latency_ms:.0f}ms"
                )
                pending = None
                reward_since_decision = 0.0
                decision_updated = True
                previous_latency_ms = active_decision.latency_ms
                previous_response_delay_frames = frame_index - pending_request_frame

            run_ended = bool(
                snapshot.dead
                or snapshot.clear
                or terminated
                or truncated
                or (next_index >= max_decisions and pending is None)
            )

            if (
                not run_ended
                and pending is None
                and next_index < max_decisions
                and frame_index - last_request_frame >= frames_per_decision
            ):
                pending_snapshot = snapshot
                pending_index = next_index
                pending = executor.submit(policy.choose, snapshot, actions)
                pending_request_frame = frame_index
                next_index += 1
                last_request_frame = frame_index

            if not run_ended:
                action = active_decision.action if active_decision else Action.NOOP
                if decision_updated and action in FIRE_ACTIONS and previous_action in FIRE_ACTIONS:
                    action = FIRE_RELEASE_ACTION[action]
                frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
                previous_action = action
                previous_reward = float(reward)
                reward_since_decision += float(reward)
                episode_reward += float(reward)
                frame_index += 1

            if audio is not None:
                audio.update(snapshot, ram=_unwrap_ram(env))

            command = dashboard.draw(
                frame,
                snapshot,
                active_decision,
                decision_index=max(0, next_index - 1),
                episode_reward=episode_reward,
                waiting=pending is not None,
                run_ended=run_ended,
                sound_enabled=(not audio.muted) if audio is not None else None,
            )
            if screenshot_path is not None and active_decision is not None and not screenshot_saved:
                dashboard.save(screenshot_path)
                screenshot_saved = True
            if command == DashboardCommand.QUIT:
                break
            if command == DashboardCommand.TOGGLE_MUTE and audio is not None:
                audio.toggle_mute()
            if command == DashboardCommand.RESTART:
                if pending is not None:
                    pending.cancel()
                frame, info = env.reset()
                parser.reset()
                if audio is not None:
                    audio.reset()
                active_decision = None
                pending = None
                pending_snapshot = None
                pending_index = 0
                pending_request_frame = 0
                next_index = 0
                frame_index = 0
                last_request_frame = -frames_per_decision
                episode_reward = 0.0
                reward_since_decision = 0.0
                previous_action = None
                previous_reward = 0.0
                previous_latency_ms = 0.0
                previous_response_delay_frames = 0
                terminated = truncated = False
                print("--- Restarted ---")


def run_episode(
    *,
    rom_path: str = DEFAULT_ROM_PATH,
    policy: Policy,
    frames_per_decision: int,
    max_decisions: int,
    seed: int,
    artifacts_dir: Path,
    display: str = "dashboard",
    screenshot_path: Path | None = None,
    sound: bool = True,
    volume: float = 0.5,
) -> Path:
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")
    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    if not 0.0 <= volume <= 1.0:
        raise ValueError("volume must be between 0.0 and 1.0")

    render_mode = "human" if display == "game" else "rgb_array"
    env = create_galaga_env(rom_path=rom_path, render_mode=render_mode)
    dashboard = LiveDashboard() if display == "dashboard" else None
    audio = (
        GalagaAudioEngine(AudioConfig(enabled=True, volume=volume, rom_path=rom_path))
        if sound and display != "none"
        else None
    )
    parser = GalagaStateParser(decision_horizon_frames=frames_per_decision)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = artifacts_dir / f"run-{timestamp}.jsonl"

    frame, info = env.reset(seed=seed)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            if dashboard:
                _run_realtime_dashboard(
                    env=env,
                    dashboard=dashboard,
                    policy=policy,
                    parser=parser,
                    frame=frame,
                    info=info,
                    log=log,
                    frames_per_decision=frames_per_decision,
                    max_decisions=max_decisions,
                    screenshot_path=screenshot_path,
                    audio=audio,
                )
                return log_path

            previous_action: Action | None = None
            previous_reward = 0.0
            previous_latency_ms = 0.0
            actions = tuple(Action)
            for decision_index in range(max_decisions):
                snapshot = parser.parse(
                    info,
                    _unwrap_ram(env),
                    previous_action=previous_action.value if previous_action else None,
                    previous_reward=previous_reward,
                    response_latency_ms=previous_latency_ms,
                    response_delay_frames=0,
                )
                if audio is not None:
                    audio.update(snapshot)
                if snapshot.dead or snapshot.clear:
                    break

                decision = policy.choose(snapshot, actions)
                action_index = ACTION_TO_INDEX[decision.action]
                total_reward = 0.0
                terminated = truncated = False
                for step_idx in range(frames_per_decision):
                    frame_action = action_index
                    if (
                        step_idx == 0
                        and decision.action in FIRE_ACTIONS
                        and previous_action in FIRE_ACTIONS
                        and frames_per_decision > 1
                    ):
                        frame_action = ACTION_TO_INDEX[FIRE_RELEASE_ACTION[decision.action]]
                    frame, reward, terminated, truncated, info = env.step(frame_action)
                    if audio is not None:
                        audio.sync_ram(_unwrap_ram(env))
                    total_reward += float(reward)
                    if terminated or truncated:
                        break

                _record_decision(
                    log,
                    decision_index=decision_index,
                    snapshot=snapshot,
                    decision=decision,
                    reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                print(
                    f"#{decision_index:04d} x={snapshot.x:03d} score={snapshot.score:06d} "
                    f"action={decision.action.value:<12} "
                    f"confidence={decision.confidence:.2f} latency={decision.latency_ms:.0f}ms"
                )

                previous_action = decision.action
                previous_reward = total_reward
                previous_latency_ms = decision.latency_ms
                if terminated or truncated:
                    break
    finally:
        if audio is not None:
            audio.close()
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
