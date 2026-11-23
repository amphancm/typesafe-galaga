from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO

from ..state import GalagaEnemy, GalagaProjectile, GalagaSnapshot
from .actions import NUM_ACTIONS, RL_ACTIONS, index_from_action
from .observations import OBSERVATION_DIM, encode_observation


@dataclass(frozen=True)
class BCDataset:
    observations: np.ndarray
    actions: np.ndarray
    action_probs: np.ndarray
    confidences: np.ndarray
    source_files: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class BCTrainResult:
    model_path: Path
    samples: int
    epochs: int
    final_train_loss: float
    final_train_accuracy: float
    val_accuracy: float | None


def _int_or(value: Any, default: int = 0) -> int:
    return int(value) if value is not None else default


def _opt_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def record_to_snapshot(record: Mapping[str, Any]) -> GalagaSnapshot:
    """Reconstruct a GalagaSnapshot from a Jev JSONL run record."""
    state = record.get("state", {})
    debug_state = record.get("debug_state", {})
    player = state.get("player", {})
    tactical = state.get("tactical", {})
    hazards = state.get("hazards", {})
    reaction = state.get("reaction_timing", {})
    recent = state.get("recent_control", {})
    episode = state.get("episode", {})

    enemies: list[GalagaEnemy] = []
    raw_enemies = debug_state.get("enemies", hazards.get("diving_enemies", ()))
    for item in raw_enemies:
        enemies.append(
            GalagaEnemy(
                x=_int_or(item.get("x"), 100),
                y=_int_or(item.get("y"), 60),
                dx_from_player=_int_or(item.get("dx_from_player"), 0),
                dy_from_player=_int_or(item.get("dy_from_player"), 100),
                diving=bool(item.get("diving", False)),
                aligned=bool(item.get("aligned", False)),
            )
        )

    projectiles: list[GalagaProjectile] = []
    for item in hazards.get("projectiles", ()):
        projectiles.append(
            GalagaProjectile(
                x=_int_or(item.get("x"), 100),
                y=_int_or(item.get("y"), 120),
                dx_from_player=_int_or(item.get("dx_from_player"), 0),
                dy_from_player=_int_or(item.get("dy_from_player"), 80),
            )
        )

    bullets = tuple(
        (_int_or(b.get("x"), 100), _int_or(b.get("y"), 180))
        for b in debug_state.get("player_bullets", ())
    )
    radar_grid = tuple(str(row) for row in debug_state.get("radar_grid", ()))
    audio_state = state.get("audio", {})
    raw_audio_events = audio_state.get("events", ()) if isinstance(audio_state, Mapping) else ()
    audio_events = tuple(str(event) for event in raw_audio_events)

    return GalagaSnapshot(
        goal=str(
            state.get("goal", "Survive and destroy all Galaga enemies for the highest score.")
        ),
        stage=_int_or(episode.get("stage"), 1),
        score=_int_or(episode.get("score"), 0),
        lives=_int_or(episode.get("lives"), 3),
        x=_int_or(player.get("x"), 100),
        y=_int_or(player.get("y"), 199),
        dx=_int_or(player.get("dx"), 0),
        alive=bool(player.get("alive", True)),
        dead=bool(episode.get("dead", False)),
        clear=bool(episode.get("clear", False)),
        active_player_bullets=_int_or(player.get("active_bullets"), len(bullets)),
        player_bullets=bullets,
        enemies=tuple(enemies),
        projectiles=tuple(projectiles),
        aligned_enemy_count=_int_or(tactical.get("aligned_enemy_count"), 0),
        danger_imminent=bool(tactical.get("danger_imminent", False)),
        recommended_evade_dir=str(tactical.get("recommended_evade_dir", "none")),
        best_target_x=_opt_int(tactical.get("best_target_x")),
        radar_grid=radar_grid,
        audio_events=audio_events,
        previous_action=recent.get("previous_action"),
        previous_reward=float(recent.get("previous_reward", 0.0)),
        decision_horizon_frames=_int_or(reaction.get("action_hold_frames"), 8),
        response_delay_frames=_int_or(reaction.get("last_response_delay_frames"), 0),
        response_latency_ms=float(reaction.get("last_response_latency_ms", 0.0)),
    )


def _expand_jsonl_paths(paths: Sequence[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            resolved.append(path)
    return resolved


def load_demonstrations(
    paths: Sequence[Path],
    *,
    min_confidence: float = 0.0,
    include_audio_obs: bool = True,
) -> BCDataset:
    """Load Jev JSONL decision logs into numeric RL observation/action tensors."""
    files = _expand_jsonl_paths(paths)
    if not files:
        raise FileNotFoundError(f"No JSONL demonstration files found in: {paths}")

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    prob_list: list[np.ndarray] = []
    conf_list: list[float] = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                confidence = float(record.get("confidence", 1.0))
                if confidence < min_confidence:
                    continue
                action_raw = record.get("action")
                if not isinstance(action_raw, str):
                    continue
                try:
                    action_idx = index_from_action(action_raw)
                except ValueError:
                    continue

                snapshot = record_to_snapshot(record)
                obs = encode_observation(snapshot, include_audio=include_audio_obs)

                raw_probs = record.get("probabilities", {})
                probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
                if isinstance(raw_probs, Mapping) and raw_probs:
                    for idx, candidate in enumerate(RL_ACTIONS):
                        probs[idx] = max(0.0, float(raw_probs.get(candidate.value, 0.0)))
                prob_sum = float(probs.sum())
                if prob_sum > 0.0:
                    probs /= prob_sum
                else:
                    probs[action_idx] = 1.0

                obs_list.append(obs)
                action_list.append(action_idx)
                prob_list.append(probs)
                conf_list.append(confidence)

    if not obs_list:
        raise ValueError("No valid demonstration samples found after filtering.")

    return BCDataset(
        observations=np.stack(obs_list, axis=0).astype(np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        action_probs=np.stack(prob_list, axis=0).astype(np.float32),
        confidences=np.asarray(conf_list, dtype=np.float32),
        source_files=tuple(str(p) for p in files),
    )


class _DummyGalagaSpaceEnv(gym.Env[np.ndarray, int]):
    """Lightweight space-only Gym env for initializing an SB3 PPO policy for BC."""

    def __init__(self, obs_dim: int = OBSERVATION_DIM) -> None:
        super().__init__()
        self._obs_dim = int(obs_dim)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(self._obs_dim, dtype=np.float32), 0.0, True, False, {}


def train_bc(
    dataset: BCDataset,
    *,
    model_path: Path = Path("models/galaga_bc.zip"),
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    soft_target_weight: float = 0.5,
    val_fraction: float = 0.15,
    net_arch: tuple[int, ...] = (256, 256),
    seed: int = 42,
    device: str = "cpu",
) -> BCTrainResult:
    """Pretrain a PPO-compatible MlpPolicy using Behavior Cloning on Jev logs."""
    if len(dataset) == 0:
        raise ValueError("dataset must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    dummy_env = _DummyGalagaSpaceEnv(obs_dim=int(dataset.observations.shape[1]))
    model = PPO(
        policy="MlpPolicy",
        env=dummy_env,
        policy_kwargs={"net_arch": list(net_arch)},
        seed=seed,
        device=device,
        verbose=0,
    )
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    indices = rng.permutation(len(dataset))
    val_size = int(len(dataset) * val_fraction) if len(dataset) >= 10 else 0
    val_idx = indices[:val_size]
    train_idx = indices[val_size:] if val_size < len(dataset) else indices

    train_obs = torch.as_tensor(dataset.observations[train_idx], device=policy.device)
    train_act = torch.as_tensor(dataset.actions[train_idx], device=policy.device)
    train_probs = torch.as_tensor(dataset.action_probs[train_idx], device=policy.device)

    final_loss = 0.0
    final_acc = 0.0
    ce_weight = 1.0 - soft_target_weight
    n_train = train_obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_train, device=policy.device)
        epoch_loss_t = torch.zeros((), device=policy.device)
        correct_t = torch.zeros((), dtype=torch.long, device=policy.device)

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            obs_b = train_obs[batch_idx]
            act_b = train_act[batch_idx]
            probs_b = train_probs[batch_idx]

            dist = policy.get_distribution(obs_b)
            logits = dist.distribution.logits
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = F.nll_loss(log_probs, act_b)
            kl_loss = F.kl_div(log_probs, probs_b, reduction="batchmean")
            loss = ce_weight * ce_loss + soft_target_weight * kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = obs_b.shape[0]
            epoch_loss_t += loss.detach() * batch_n
            correct_t += (logits.detach().argmax(dim=-1) == act_b).sum()

        denom = max(1, n_train)
        final_loss = float(epoch_loss_t.item()) / denom
        final_acc = float(correct_t.item()) / denom

    val_acc: float | None = None
    if len(val_idx) > 0:
        policy.set_training_mode(False)
        with torch.inference_mode():
            val_obs = torch.as_tensor(dataset.observations[val_idx], device=policy.device)
            val_act = torch.as_tensor(dataset.actions[val_idx], device=policy.device)
            val_logits = policy.get_distribution(val_obs).distribution.logits
            val_acc = float((val_logits.argmax(dim=-1) == val_act).float().mean().item())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    dummy_env.close()

    return BCTrainResult(
        model_path=model_path,
        samples=len(dataset),
        epochs=epochs,
        final_train_loss=final_loss,
        final_train_accuracy=final_acc,
        val_accuracy=val_acc,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain a Galaga PPO policy via Behavior Cloning from Jev JSONL logs."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[Path("artifacts")],
        help="JSONL files or directories containing Jev run logs",
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/galaga_bc.zip"))
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--soft-target-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--audio-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include semantic audio event flags in BC observations (default: enabled)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_demonstrations(
        args.inputs,
        min_confidence=args.min_confidence,
        include_audio_obs=args.audio_obs,
    )
    result = train_bc(
        dataset,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        soft_target_weight=args.soft_target_weight,
        seed=args.seed,
        device=args.device,
    )
    val_str = f"{result.val_accuracy * 100:.1f}%" if result.val_accuracy is not None else "n/a"
    print(
        f"BC complete: samples={result.samples} epochs={result.epochs} "
        f"train_loss={result.final_train_loss:.4f} "
        f"train_acc={result.final_train_accuracy * 100:.1f}% val_acc={val_str} "
        f"saved={result.model_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
