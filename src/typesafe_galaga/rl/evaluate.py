from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from ..runner import DEFAULT_ROM_PATH
from .env import GalagaRLEnv
from .observations import FrameStackBuffer, infer_obs_config_from_shape, observation_dim


@dataclass(frozen=True)
class EvalSummary:
    model_path: str
    rom_path: str
    episodes: int
    deterministic: bool
    frames_per_decision: int
    random_noop_max: int
    clear_rate: float
    death_rate: float
    mean_reward: float
    std_reward: float
    mean_score: float
    max_score: int
    min_score: int
    mean_steps: float

    def to_text(self) -> str:
        return (
            f"Model:           {self.model_path}\n"
            f"ROM:             {self.rom_path}\n"
            f"Episodes:        {self.episodes} "
            f"({'deterministic' if self.deterministic else 'stochastic'})\n"
            f"Clear rate:      {self.clear_rate * 100:.1f}%\n"
            f"Death rate:      {self.death_rate * 100:.1f}%\n"
            f"Episode reward:  {self.mean_reward:.2f} +/- {self.std_reward:.2f}\n"
            f"Score:           mean={self.mean_score:.1f}, min={self.min_score}, "
            f"max={self.max_score}\n"
            f"Mean steps:      {self.mean_steps:.1f}"
        )


def evaluate_policy(
    model_path: Path,
    *,
    rom_path: str = DEFAULT_ROM_PATH,
    episodes: int = 20,
    frames_per_decision: int = 4,
    max_steps: int = 4500,
    random_noop_max: int = 4,
    deterministic: bool = True,
    seed: int = 1000,
    device: str = "cpu",
) -> EvalSummary:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    model = PPO.load(str(model_path), device=device)
    policy = model.policy
    policy.set_training_mode(False)
    policy_device = model.device
    include_audio_obs, frame_stack = infer_obs_config_from_shape(
        getattr(model.observation_space, "shape", None)
    )
    single_obs_dim = observation_dim(include_audio=include_audio_obs)
    frame_buffer = FrameStackBuffer(single_obs_dim, frame_stack)
    env = GalagaRLEnv(
        rom_path=rom_path,
        frames_per_decision=frames_per_decision,
        max_steps=max_steps,
        random_noop_max=random_noop_max,
        render_mode="rgb_array",
        include_audio_obs=include_audio_obs,
    )

    rewards: list[float] = []
    scores: list[int] = []
    clears: list[int] = []
    deaths: list[int] = []
    steps_list: list[int] = []

    try:
        with torch.inference_mode():
            for ep in range(episodes):
                obs, info = env.reset(seed=seed + ep)
                stacked_obs = frame_buffer.reset(obs)
                terminated = False
                truncated = False
                ep_reward = 0.0
                last_info = info

                while not (terminated or truncated):
                    obs_tensor = torch.as_tensor(stacked_obs, device=policy_device).unsqueeze(0)
                    action = int(policy._predict(obs_tensor, deterministic=deterministic).item())
                    obs, reward, terminated, truncated, last_info = env.step(action)
                    stacked_obs = frame_buffer.push(obs)
                    ep_reward += float(reward)

                rewards.append(ep_reward)
                scores.append(int(last_info.get("score", 0)))
                clears.append(int(bool(last_info.get("clear", False))))
                deaths.append(int(bool(last_info.get("dead", False))))
                steps_list.append(int(last_info.get("steps", 0)))
    finally:
        env.close()

    return EvalSummary(
        model_path=str(model_path),
        rom_path=rom_path,
        episodes=episodes,
        deterministic=deterministic,
        frames_per_decision=frames_per_decision,
        random_noop_max=random_noop_max,
        clear_rate=float(np.mean(clears)),
        death_rate=float(np.mean(deaths)),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_score=float(np.mean(scores)),
        max_score=max(scores),
        min_score=min(scores),
        mean_steps=float(np.mean(steps_list)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Galaga PPO policy")
    parser.add_argument("--model", type=Path, default=Path("models/galaga_ppo.zip"))
    parser.add_argument("--rom", default=DEFAULT_ROM_PATH)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frames-per-decision", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--random-noop-max", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_policy(
        args.model,
        rom_path=args.rom,
        episodes=args.episodes,
        frames_per_decision=args.frames_per_decision,
        max_steps=args.max_steps,
        random_noop_max=args.random_noop_max,
        deterministic=not args.stochastic,
        seed=args.seed,
        device=args.device,
    )
    print(summary.to_text())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
