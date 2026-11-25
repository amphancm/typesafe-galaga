from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from stable_baselines3 import PPO

from ..actions import ACTION_TO_INDEX, Action
from ..audio import AudioConfig, GalagaAudioEngine
from ..dashboard import DashboardCommand, DashboardConfig, LiveDashboard
from ..policy import Decision, Policy, TypeSafePolicy
from ..runner import DEFAULT_ROM_PATH
from ..state import GalagaSnapshot
from .actions import RL_ACTIONS, action_from_index
from .env import GalagaRLEnv
from .hybrid import HybridConfig, HybridPolicy
from .observations import (
    FrameStackBuffer,
    encode_observation,
    infer_obs_config_from_shape,
    observation_dim,
)


class RLPolicy:
    """Policy adapter wrapping a trained SB3 PPO model for LiveDashboard and runner."""

    def __init__(
        self,
        model_path: Path,
        *,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model = PPO.load(str(model_path), device=device)
        self.model.policy.set_training_mode(False)
        self._policy = self.model.policy
        self._device = self.model.device
        self.deterministic = deterministic
        self.include_audio_obs, self.frame_stack = infer_obs_config_from_shape(
            getattr(self.model.observation_space, "shape", None)
        )
        single_obs_dim = observation_dim(include_audio=self.include_audio_obs)
        self._frame_buffer = FrameStackBuffer(single_obs_dim, self.frame_stack)

    def reset_episode(self) -> None:
        self._frame_buffer.reset()

    def choose(self, snapshot: GalagaSnapshot, actions: Sequence[Action]) -> Decision:
        del actions
        started = time.perf_counter()
        obs = encode_observation(snapshot, include_audio=self.include_audio_obs)
        stacked_obs = self._frame_buffer.push(obs)
        obs_tensor = torch.as_tensor(stacked_obs, device=self._device).unsqueeze(0)
        with torch.inference_mode():
            dist = self._policy.get_distribution(obs_tensor)
            probs_tensor = dist.distribution.probs.squeeze(0)
            if self.deterministic:
                action_idx = int(probs_tensor.argmax().item())
            else:
                action_idx = int(dist.sample().item())
            probs_list = probs_tensor.cpu().tolist()

        action = action_from_index(action_idx)
        probabilities = {
            candidate.value: probs_list[idx] for idx, candidate in enumerate(RL_ACTIONS)
        }
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Decision(
            action=action,
            confidence=probs_list[action_idx],
            probabilities=probabilities,
            latency_ms=latency_ms,
        )


def play_rl(
    model_path: Path,
    *,
    rom_path: str = DEFAULT_ROM_PATH,
    frames_per_decision: int = 4,
    max_steps: int = 4500,
    deterministic: bool = True,
    hybrid: bool = False,
    confidence_threshold: float = 0.55,
    display: str = "dashboard",
    speed: float = 1.0,
    sound: bool = True,
    volume: float = 0.5,
    seed: int = 123,
    device: str = "cpu",
) -> dict[str, object]:
    """Run a trained RL agent (or Hybrid RL + Jev Strategic Agent) on NES Galaga."""
    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    if speed <= 0:
        raise ValueError("speed must be greater than 0")
    if not 0.0 <= volume <= 1.0:
        raise ValueError("volume must be between 0.0 and 1.0")

    rl_policy = RLPolicy(model_path, deterministic=deterministic, device=device)
    hybrid_policy: HybridPolicy | None = None
    active_policy: Policy = rl_policy
    if hybrid:
        hybrid_policy = HybridPolicy(
            rl_policy,
            TypeSafePolicy(),
            config=HybridConfig(confidence_threshold=confidence_threshold),
        )
        active_policy = hybrid_policy

    render_mode = "human" if display == "game" else "rgb_array"
    use_audio = sound and display != "none"
    env = GalagaRLEnv(
        rom_path=rom_path,
        frames_per_decision=frames_per_decision,
        max_steps=max_steps,
        render_mode=render_mode,
        capture_step_frames=(display == "dashboard"),
        capture_audio_events=use_audio,
        include_audio_obs=getattr(rl_policy, "include_audio_obs", True),
    )
    target_fps = max(1, round(60 * speed))
    dashboard = LiveDashboard(DashboardConfig(fps=target_fps)) if display == "dashboard" else None
    audio = (
        GalagaAudioEngine(AudioConfig(enabled=True, volume=volume, rom_path=rom_path))
        if use_audio
        else None
    )

    try:
        _, info = env.reset(seed=seed)
        if hasattr(rl_policy, "reset_episode"):
            rl_policy.reset_episode()
        if hybrid_policy is not None:
            hybrid_policy.reset_episode()
        if audio is not None:
            audio.reset()
            audio.update(env.last_snapshot, area_type=getattr(env, "area_type", None))
        terminated = False
        truncated = False
        clear_animating = False
        episode_reward = 0.0
        decision_index = 0
        last_decision: Decision | None = None

        while True:
            step_started = time.perf_counter()
            snapshot = env.last_snapshot
            run_ended = terminated or truncated
            frames_to_draw: tuple[object, ...] = ()
            step_audio_events: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
            frames_executed = 0

            if not run_ended and snapshot is not None:
                last_decision = active_policy.choose(snapshot, RL_ACTIONS)
                action_idx = ACTION_TO_INDEX[last_decision.action]
                _, reward, terminated, truncated, info = env.step(action_idx)
                episode_reward += float(reward)
                decision_index += 1
                frames_executed = int(info.get("frames_executed", frames_per_decision))
                snapshot = env.last_snapshot
                run_ended = terminated or truncated
                if (
                    run_ended
                    and bool(info.get("clear", False))
                    and display in {"dashboard", "game"}
                ):
                    clear_animating = callable(getattr(env, "step_clear_animation", None))
                step_audio_events = tuple(getattr(env, "last_step_audio_events", ()))
                if audio is not None and not step_audio_events:
                    audio.update(
                        snapshot,
                        area_type=getattr(env, "area_type", None),
                        finishing=clear_animating,
                    )
                elif audio is not None and dashboard is None:
                    for started, stopped in step_audio_events:
                        audio.apply_rom_events(started, stopped)
                if dashboard is not None:
                    frames_to_draw = env.last_step_frames
            elif clear_animating:
                step_anim = getattr(env, "step_clear_animation", None)
                anim_frame = step_anim() if callable(step_anim) else None
                if anim_frame is not None:
                    snapshot = env.last_snapshot
                    frames_executed = 1
                    if audio is not None:
                        audio.update(
                            snapshot,
                            area_type=getattr(env, "area_type", None),
                            finishing=True,
                        )
                    if dashboard is not None:
                        frames_to_draw = (anim_frame,)
                    elif display == "game":
                        env.render()
                else:
                    clear_animating = False

            if dashboard is not None and snapshot is not None:
                if not frames_to_draw:
                    frames_to_draw = (env.last_frame,)
                should_quit = False
                for frame_idx, frame in enumerate(frames_to_draw):
                    if audio is not None and frame_idx < len(step_audio_events):
                        started, stopped = step_audio_events[frame_idx]
                        audio.apply_rom_events(started, stopped)
                    command = dashboard.draw(
                        frame,
                        snapshot,
                        last_decision,
                        decision_index=decision_index,
                        episode_reward=episode_reward,
                        waiting=False,
                        run_ended=run_ended,
                        finishing=clear_animating,
                        sound_enabled=(not audio.muted) if audio is not None else None,
                    )
                    if command == DashboardCommand.QUIT:
                        should_quit = True
                        break
                    if command == DashboardCommand.TOGGLE_MUTE and audio is not None:
                        audio.toggle_mute()
                    if command == DashboardCommand.RESTART:
                        _, info = env.reset(seed=seed)
                        if hasattr(rl_policy, "reset_episode"):
                            rl_policy.reset_episode()
                        if hybrid_policy is not None:
                            hybrid_policy.reset_episode()
                        if audio is not None:
                            audio.reset()
                            audio.update(
                                env.last_snapshot,
                                area_type=getattr(env, "area_type", None),
                            )
                        terminated = False
                        truncated = False
                        clear_animating = False
                        episode_reward = 0.0
                        decision_index = 0
                        last_decision = None
                        break
                if should_quit:
                    break
            elif run_ended and not clear_animating:
                break
            elif display == "game" and frames_executed > 0:
                target_duration = frames_executed / (60.0 * speed)
                remaining = target_duration - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)

        summary: dict[str, object] = {
            "clear": bool(info.get("clear", False)),
            "dead": bool(info.get("dead", False)),
            "score": int(info.get("score", 0)),
            "steps": decision_index,
            "episode_reward": episode_reward,
        }
        if hybrid_policy is not None:
            hstats = hybrid_policy.stats()
            summary["rl_decisions"] = hstats.rl_decisions
            summary["jev_decisions"] = hstats.jev_decisions
            summary["api_reduction_rate"] = hstats.api_reduction_rate
        return summary
    finally:
        if audio is not None:
            audio.close()
        if dashboard is not None:
            dashboard.close()
        if hybrid_policy is not None:
            hybrid_policy.close()
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play NES Galaga with a trained RL policy.")
    parser.add_argument("--model", type=Path, default=Path("models/galaga_ppo.zip"))
    parser.add_argument("--rom", default=DEFAULT_ROM_PATH)
    parser.add_argument("--frames-per-decision", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--display", choices=("dashboard", "game", "none"), default="dashboard")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier relative to 60 FPS NES speed (1.0 = normal)",
    )
    parser.add_argument(
        "--sound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NES ROM music and sound effects (use --no-sound to mute)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Master audio volume from 0.0 to 1.0 (default: 0.5)",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable Hybrid AI (RL fast control + Jev Strategic Agent on low confidence/danger)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = play_rl(
        args.model,
        rom_path=args.rom,
        frames_per_decision=args.frames_per_decision,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        hybrid=args.hybrid,
        confidence_threshold=args.confidence_threshold,
        display=args.display,
        speed=args.speed,
        sound=args.sound,
        volume=args.volume,
        seed=args.seed,
        device=args.device,
    )
    msg = (
        f"RL play finished: clear={summary['clear']} dead={summary['dead']} "
        f"score={summary['score']} steps={summary['steps']} "
        f"reward={summary['episode_reward']:.2f}"
    )
    if "jev_decisions" in summary:
        msg += (
            f" rl_calls={summary['rl_decisions']} jev_calls={summary['jev_decisions']} "
            f"api_reduction={float(summary['api_reduction_rate']) * 100:.1f}%"
        )
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
