from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)

from ..runner import DEFAULT_ROM_PATH
from .env import GalagaRLEnv


@dataclass(frozen=True)
class TrainConfig:
    rom_path: str = DEFAULT_ROM_PATH
    total_timesteps: int = 5_000_000
    num_envs: int = 8
    frames_per_decision: int = 4
    max_steps: int = 4500
    random_noop_max: int = 4
    frame_stack: int = 4
    normalize_reward: bool = True
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    pi_arch: tuple[int, ...] = (256, 256)
    vf_arch: tuple[int, ...] = (512, 512)
    seed: int = 42
    device: str = "cpu"
    model_path: Path = Path("models/galaga_ppo.zip")
    checkpoint_dir: Path = Path("models/checkpoints")
    checkpoint_freq: int = 100_000
    eval_freq: int = 200_000
    eval_episodes: int = 10
    tensorboard_log: Path = Path("runs/galaga_ppo")
    resume_from: Path | None = None
    target_clear_rate: float | None = None
    include_audio_obs: bool = True


def _linear_schedule(initial_value: float, min_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return schedule


def _make_env(
    rom_path: str,
    *,
    frames_per_decision: int,
    max_steps: int,
    random_noop_max: int,
    include_audio_obs: bool,
    seed: int,
    rank: int,
) -> Callable[[], Any]:
    def _init() -> Any:
        env = GalagaRLEnv(
            rom_path=rom_path,
            frames_per_decision=frames_per_decision,
            max_steps=max_steps,
            random_noop_max=random_noop_max,
            render_mode="rgb_array",
            include_audio_obs=include_audio_obs,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


class GalagaProgressCallback(BaseCallback):
    """Track Galaga episode score/survival metrics and optionally persist the best model."""

    def __init__(
        self,
        *,
        best_model_path: Path,
        window_size: int = 100,
        log_interval_steps: int = 20_000,
        target_clear_rate: float | None = None,
        save_best: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.best_model_path = best_model_path
        self.window_size = window_size
        self.log_interval_steps = log_interval_steps
        self.target_clear_rate = target_clear_rate
        self.save_best = save_best
        self._recent_scores: deque[int] = deque(maxlen=window_size)
        self._recent_clears: deque[int] = deque(maxlen=window_size)
        self._recent_deaths: deque[int] = deque(maxlen=window_size)
        self._recent_rewards: deque[float] = deque(maxlen=window_size)
        self._best_mean_score = -1.0
        self._last_log_step = 0
        self._total_episodes = 0
        self._total_clears = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", ())
        infos = self.locals.get("infos", ())
        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue
            self._total_episodes += 1
            score = int(info.get("score", 0))
            cleared = int(bool(info.get("clear", False)))
            died = int(bool(info.get("dead", False)))
            ep_reward = float(info.get("episode_reward", 0.0))

            self._recent_scores.append(score)
            self._recent_clears.append(cleared)
            self._recent_deaths.append(died)
            self._recent_rewards.append(ep_reward)
            self._total_clears += cleared

        if (
            self._recent_scores
            and (self.num_timesteps - self._last_log_step) >= self.log_interval_steps
        ):
            self._last_log_step = self.num_timesteps
            mean_score = sum(self._recent_scores) / len(self._recent_scores)
            max_score = max(self._recent_scores)
            clear_rate = sum(self._recent_clears) / len(self._recent_clears)
            death_rate = sum(self._recent_deaths) / len(self._recent_deaths)
            mean_rew = sum(self._recent_rewards) / len(self._recent_rewards)

            self.logger.record("galaga/mean_score", mean_score)
            self.logger.record("galaga/max_score", float(max_score))
            self.logger.record("galaga/clear_rate", clear_rate)
            self.logger.record("galaga/death_rate", death_rate)
            self.logger.record("galaga/mean_episode_reward", mean_rew)

            if (
                self.save_best
                and mean_score > self._best_mean_score
                and len(self._recent_scores) >= 10
            ):
                self._best_mean_score = mean_score
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))

            if self.verbose:
                print(
                    f"[step={self.num_timesteps:>8d}] eps={self._total_episodes:>5d} "
                    f"mean_score={mean_score:>7.1f} max_score={max_score:>6d} "
                    f"death={death_rate * 100:>5.1f}% rew={mean_rew:>7.1f}",
                    flush=True,
                )

            if (
                self.target_clear_rate is not None
                and len(self._recent_clears) >= min(50, self.window_size)
                and clear_rate >= self.target_clear_rate
            ):
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
                if self.verbose:
                    print(
                        f"Early stopping: reached target clear rate "
                        f"{clear_rate * 100:.1f}% >= {self.target_clear_rate * 100:.1f}%",
                        flush=True,
                    )
                return False
        return True


class GalagaEvalCallback(BaseCallback):
    """Run periodic deterministic evaluation episodes and save the highest-scoring model."""

    def __init__(
        self,
        *,
        eval_env: Any,
        best_model_path: Path,
        eval_freq: int = 200_000,
        n_eval_episodes: int = 10,
        eval_seed: int = 1000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.best_model_path = best_model_path
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.eval_seed = int(eval_seed)
        self._last_eval_step = 0
        self._best_eval_score = -1.0

    def _run_eval(self) -> tuple[float, int, float, float]:
        policy = self.model.policy
        was_training = policy.training
        policy.set_training_mode(False)
        device = self.model.device

        scores: list[int] = []
        rewards: list[float] = []
        steps_list: list[int] = []
        try:
            with torch.inference_mode():
                for ep in range(self.n_eval_episodes):
                    self.eval_env.seed(self.eval_seed + ep)
                    obs = self.eval_env.reset()
                    done = False
                    last_info: dict[str, Any] = {}
                    while not done:
                        obs_tensor = torch.as_tensor(obs, device=device)
                        action = int(policy._predict(obs_tensor, deterministic=True).item())
                        obs, _, dones, infos = self.eval_env.step(np.array([action]))
                        done = bool(dones[0])
                        last_info = infos[0]
                    scores.append(int(last_info.get("score", 0)))
                    rewards.append(float(last_info.get("episode_reward", 0.0)))
                    steps_list.append(int(last_info.get("steps", 0)))
        finally:
            policy.set_training_mode(was_training)

        return (
            float(np.mean(scores)),
            max(scores),
            float(np.mean(rewards)),
            float(np.mean(steps_list)),
        )

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        mean_score, max_score, mean_rew, mean_steps = self._run_eval()
        self.logger.record("eval/mean_score", mean_score)
        self.logger.record("eval/max_score", float(max_score))
        self.logger.record("eval/mean_episode_reward", mean_rew)
        self.logger.record("eval/mean_steps", mean_steps)

        is_best = mean_score > self._best_eval_score
        if is_best:
            self._best_eval_score = mean_score
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_path))

        if self.verbose:
            marker = " *NEW BEST*" if is_best else ""
            print(
                f"[eval step={self.num_timesteps:>8d}] eps={self.n_eval_episodes} "
                f"mean_score={mean_score:>7.1f} max_score={max_score:>6d} "
                f"rew={mean_rew:>7.1f} steps={mean_steps:>6.1f}{marker}",
                flush=True,
            )
        return True


def train_ppo(config: TrainConfig) -> Path:
    """Train a PPO agent headless on NES Galaga."""
    if config.frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_log.mkdir(parents=True, exist_ok=True)

    env_fns = [
        _make_env(
            config.rom_path,
            frames_per_decision=config.frames_per_decision,
            max_steps=config.max_steps,
            random_noop_max=config.random_noop_max,
            include_audio_obs=config.include_audio_obs,
            seed=config.seed,
            rank=rank,
        )
        for rank in range(config.num_envs)
    ]
    vec_env: Any = SubprocVecEnv(env_fns) if config.num_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    if config.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=config.frame_stack)
    if config.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=config.gamma,
        )

    eval_env: Any | None = None
    if config.eval_freq > 0 and config.eval_episodes > 0:
        eval_env = DummyVecEnv(
            [
                _make_env(
                    config.rom_path,
                    frames_per_decision=config.frames_per_decision,
                    max_steps=config.max_steps,
                    random_noop_max=config.random_noop_max,
                    include_audio_obs=config.include_audio_obs,
                    seed=config.seed + 10_000,
                    rank=0,
                )
            ]
        )
        if config.frame_stack > 1:
            eval_env = VecFrameStack(eval_env, n_stack=config.frame_stack)

    try:
        if config.resume_from is not None:
            try:
                model = PPO.load(
                    str(config.resume_from),
                    env=vec_env,
                    device=config.device,
                    tensorboard_log=str(config.tensorboard_log),
                    custom_objects={
                        "learning_rate": _linear_schedule(
                            config.learning_rate, config.min_learning_rate
                        ),
                        "ent_coef": config.ent_coef,
                    },
                )
            except ValueError as exc:
                if "Observation spaces do not match" in str(exc):
                    hint = (
                        "Ensure --audio-obs/--no-audio-obs and --frame-stack match the saved "
                        "checkpoint, or omit --resume-from to train a new model from scratch."
                    )
                    raise ValueError(f"{exc}. {hint}") from exc
                raise
        else:
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=_linear_schedule(config.learning_rate, config.min_learning_rate),
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                policy_kwargs={
                    "net_arch": {
                        "pi": list(config.pi_arch),
                        "vf": list(config.vf_arch),
                    }
                },
                tensorboard_log=str(config.tensorboard_log),
                seed=config.seed,
                device=config.device,
                verbose=0,
            )

        best_model_path = config.model_path.with_name(f"{config.model_path.stem}_best.zip")
        obs_dim = vec_env.observation_space.shape[0]
        print(
            f"Training PPO on device={model.device} with {config.num_envs} envs "
            f"(obs_dim={obs_dim}, actions={vec_env.action_space.n}, "
            f"frame_stack={config.frame_stack}, audio_obs={config.include_audio_obs}, "
            f"norm_reward={config.normalize_reward})",
            flush=True,
        )
        callbacks: list[BaseCallback] = [
            GalagaProgressCallback(
                best_model_path=best_model_path,
                target_clear_rate=config.target_clear_rate,
                save_best=(eval_env is None),
                verbose=1,
            ),
        ]
        if eval_env is not None:
            callbacks.append(
                GalagaEvalCallback(
                    eval_env=eval_env,
                    best_model_path=best_model_path,
                    eval_freq=config.eval_freq,
                    n_eval_episodes=config.eval_episodes,
                    verbose=1,
                )
            )
        if config.checkpoint_freq > 0:
            save_freq = max(1, config.checkpoint_freq // max(1, config.num_envs))
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(config.checkpoint_dir),
                    name_prefix=config.model_path.stem,
                )
            )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        model.save(str(config.model_path))
        print(f"Saved final PPO model to {config.model_path.resolve()}", flush=True)
        if best_model_path.exists():
            print(f"Best checkpoint saved at {best_model_path.resolve()}", flush=True)
        return config.model_path
    finally:
        if eval_env is not None:
            eval_env.close()
        vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO agent on NES Galaga.")
    parser.add_argument("--rom", default=DEFAULT_ROM_PATH)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frames-per-decision", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--random-noop-max", type=int, default=4)
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of consecutive observation steps to stack (default: 4, 1 to disable)",
    )
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize rewards via VecNormalize during PPO training (default: enabled)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", type=Path, default=Path("models/galaga_ppo.zip"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=200_000,
        help="Run deterministic evaluation every N timesteps (0 to disable, default: 200000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of deterministic episodes per evaluation (default: 10)",
    )
    parser.add_argument("--tensorboard-log", type=Path, default=Path("runs/galaga_ppo"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--target-clear-rate",
        type=float,
        default=None,
        help="Optional early-stop rolling clear rate in [0, 1] (e.g. 0.90)",
    )
    parser.add_argument(
        "--audio-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include semantic audio event flags in RL observations (default: enabled)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = TrainConfig(
        rom_path=args.rom,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        frames_per_decision=args.frames_per_decision,
        max_steps=args.max_steps,
        random_noop_max=args.random_noop_max,
        frame_stack=args.frame_stack,
        normalize_reward=args.normalize_reward,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        tensorboard_log=args.tensorboard_log,
        resume_from=args.resume_from,
        target_clear_rate=args.target_clear_rate,
        include_audio_obs=args.audio_obs,
    )
    train_ppo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
