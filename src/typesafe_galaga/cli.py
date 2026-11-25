from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .policy import HeuristicPolicy, TypeSafePolicy
from .runner import DEFAULT_ROM_PATH, run_episode
from .state import (
    OAM_START,
    PLAYER_X_ADDR,
    PLAYER_Y,
    SCORE_DIGITS_ADDR,
    TILE_PLAYER_BULLET,
    TILE_PLAYER_SHIP,
    GalagaStateParser,
)


def _demo_ram() -> bytearray:
    ram = bytearray(0x0800)
    ram[PLAYER_X_ADDR] = 100
    ram[SCORE_DIGITS_ADDR + 4] = 2
    ram[SCORE_DIGITS_ADDR + 5] = 6

    # Player ship (two 8px halves at y=199)
    ram[OAM_START : OAM_START + 4] = bytes((PLAYER_Y, TILE_PLAYER_SHIP, 0x20, 96))
    ram[OAM_START + 4 : OAM_START + 8] = bytes((PLAYER_Y, TILE_PLAYER_SHIP, 0x60, 103))
    # One active player bullet
    ram[OAM_START + 8 : OAM_START + 12] = bytes((140, TILE_PLAYER_BULLET, 0x03, 100))
    # One formation enemy and one diving enemy
    ram[OAM_START + 12 : OAM_START + 16] = bytes((52, 0x5E, 0xC1, 96))
    ram[OAM_START + 16 : OAM_START + 20] = bytes((52, 0x5C, 0xC1, 104))
    ram[OAM_START + 20 : OAM_START + 24] = bytes((150, 0x42, 0x80, 112))
    ram[OAM_START + 24 : OAM_START + 28] = bytes((150, 0x44, 0x80, 120))
    # One enemy projectile
    ram[OAM_START + 28 : OAM_START + 32] = bytes((165, 0xF4, 0x01, 102))
    return ram


def state_demo() -> int:
    info = {
        "stage": 1,
        "score": 260,
        "lives": 3,
        "death": False,
        "clear": False,
    }
    snapshot = GalagaStateParser().parse(info, _demo_ram(), previous_action="fire")
    print(json.dumps(snapshot.to_state(), indent=2))
    print("\n--- TEXT VIEW ---\n")
    print(snapshot.to_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="typesafe-galaga")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state-demo", help="Show structured and text state without API calls")

    play = subparsers.add_parser("play", help="Run a Galaga episode")
    play.add_argument("--rom", default=DEFAULT_ROM_PATH, help="Path to Galaga NES ROM")
    play.add_argument(
        "--frames-per-decision",
        type=int,
        default=8,
        help="Minimum macro duration in emulator frames",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--policy", choices=("typesafe", "heuristic"), default="typesafe")
    play.add_argument(
        "--display",
        choices=("dashboard", "game", "none"),
        default="dashboard",
        help="Combined telemetry dashboard, plain game window, or headless mode",
    )
    play.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    play.add_argument(
        "--screenshot",
        type=Path,
        help="Save the first populated dashboard frame as a PNG",
    )
    play.add_argument(
        "--sound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NES ROM music and sound effects (use --no-sound to mute)",
    )
    play.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Master audio volume from 0.0 to 1.0 (default: 0.5)",
    )

    rl_bc = subparsers.add_parser(
        "rl-bc",
        help="Pretrain a PPO policy via Behavior Cloning from Jev JSONL logs",
    )
    rl_bc.add_argument("--inputs", type=Path, nargs="+", default=[Path("artifacts")])
    rl_bc.add_argument("--model-path", type=Path, default=Path("models/galaga_bc.zip"))
    rl_bc.add_argument("--min-confidence", type=float, default=0.0)
    rl_bc.add_argument("--epochs", type=int, default=30)
    rl_bc.add_argument("--device", default="cpu")
    rl_bc.add_argument(
        "--audio-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include semantic audio event flags in BC observations (default: enabled)",
    )
    rl_bc.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Number of consecutive observation steps to stack (default: 1)",
    )

    rl_train = subparsers.add_parser("rl-train", help="Train a PPO agent on NES Galaga")
    rl_train.add_argument("--rom", default=DEFAULT_ROM_PATH)
    rl_train.add_argument("--timesteps", type=int, default=5_000_000)
    rl_train.add_argument("--num-envs", type=int, default=8)
    rl_train.add_argument("--frames-per-decision", type=int, default=4)
    rl_train.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of consecutive observation steps to stack (default: 4, 1 to disable)",
    )
    rl_train.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize rewards via VecNormalize during PPO training (default: enabled)",
    )
    rl_train.add_argument("--gamma", type=float, default=0.995)
    rl_train.add_argument("--eval-freq", type=int, default=200_000)
    rl_train.add_argument("--eval-episodes", type=int, default=10)
    rl_train.add_argument("--model-path", type=Path, default=Path("models/galaga_ppo.zip"))
    rl_train.add_argument("--resume-from", type=Path, default=None)
    rl_train.add_argument("--device", default="cpu")
    rl_train.add_argument(
        "--audio-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include semantic audio event flags in RL observations (default: enabled)",
    )

    rl_eval = subparsers.add_parser("rl-eval", help="Evaluate a trained Galaga PPO policy")
    rl_eval.add_argument("--model", type=Path, default=Path("models/galaga_ppo.zip"))
    rl_eval.add_argument("--rom", default=DEFAULT_ROM_PATH)
    rl_eval.add_argument("--episodes", type=int, default=20)
    rl_eval.add_argument("--frames-per-decision", type=int, default=4)
    rl_eval.add_argument("--random-noop-max", type=int, default=4)
    rl_eval.add_argument("--stochastic", action="store_true")
    rl_eval.add_argument("--device", default="cpu")
    rl_eval.add_argument("--output", type=Path, default=None)

    rl_play = subparsers.add_parser("rl-play", help="Play Galaga using a trained PPO model")
    rl_play.add_argument("--model", type=Path, default=Path("models/galaga_ppo.zip"))
    rl_play.add_argument("--rom", default=DEFAULT_ROM_PATH)
    rl_play.add_argument("--frames-per-decision", type=int, default=4)
    rl_play.add_argument("--display", choices=("dashboard", "game", "none"), default="dashboard")
    rl_play.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier relative to 60 FPS NES speed (1.0 = normal)",
    )
    rl_play.add_argument(
        "--sound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NES ROM music and sound effects (use --no-sound to mute)",
    )
    rl_play.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Master audio volume from 0.0 to 1.0 (default: 0.5)",
    )
    rl_play.add_argument("--stochastic", action="store_true")
    rl_play.add_argument("--hybrid", action="store_true")
    rl_play.add_argument("--confidence-threshold", type=float, default=0.55)
    rl_play.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "state-demo":
        return state_demo()
    if args.command == "play":
        policy = TypeSafePolicy() if args.policy == "typesafe" else HeuristicPolicy()
        log_path = run_episode(
            rom_path=args.rom,
            policy=policy,
            frames_per_decision=args.frames_per_decision,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            display=args.display,
            screenshot_path=args.screenshot,
            sound=args.sound,
            volume=args.volume,
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    if args.command == "rl-bc":
        from .rl.bc import load_demonstrations, train_bc

        dataset = load_demonstrations(
            args.inputs,
            min_confidence=args.min_confidence,
            include_audio_obs=args.audio_obs,
            frame_stack=args.frame_stack,
        )
        result = train_bc(
            dataset,
            model_path=args.model_path,
            epochs=args.epochs,
            device=args.device,
        )
        val_str = f"{result.val_accuracy * 100:.1f}%" if result.val_accuracy is not None else "n/a"
        print(
            f"BC complete: samples={result.samples} train_acc={result.final_train_accuracy * 100:.1f}% "
            f"val_acc={val_str} saved={result.model_path.resolve()}"
        )
        return 0
    if args.command == "rl-train":
        from .rl.train import TrainConfig, train_ppo

        train_ppo(
            TrainConfig(
                rom_path=args.rom,
                total_timesteps=args.timesteps,
                num_envs=args.num_envs,
                frames_per_decision=args.frames_per_decision,
                frame_stack=args.frame_stack,
                normalize_reward=args.normalize_reward,
                gamma=args.gamma,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                model_path=args.model_path,
                resume_from=args.resume_from,
                device=args.device,
                include_audio_obs=args.audio_obs,
            )
        )
        return 0
    if args.command == "rl-eval":
        from .rl.evaluate import main as eval_main

        eval_argv = [
            "--model",
            str(args.model),
            "--rom",
            args.rom,
            "--episodes",
            str(args.episodes),
            "--frames-per-decision",
            str(args.frames_per_decision),
            "--random-noop-max",
            str(args.random_noop_max),
            "--device",
            args.device,
        ]
        if args.stochastic:
            eval_argv.append("--stochastic")
        if args.output is not None:
            eval_argv.extend(["--output", str(args.output)])
        return eval_main(eval_argv)
    if args.command == "rl-play":
        from .rl.play import play_rl

        summary = play_rl(
            args.model,
            rom_path=args.rom,
            frames_per_decision=args.frames_per_decision,
            deterministic=not args.stochastic,
            hybrid=args.hybrid,
            confidence_threshold=args.confidence_threshold,
            display=args.display,
            speed=args.speed,
            sound=args.sound,
            volume=args.volume,
            device=args.device,
        )
        print(
            f"RL play finished: clear={summary['clear']} dead={summary['dead']} "
            f"score={summary['score']} steps={summary['steps']} "
            f"reward={summary['episode_reward']:.2f}"
        )
        if "jev_decisions" in summary:
            print(
                f"Hybrid stats: rl_decisions={summary['rl_decisions']} "
                f"jev_decisions={summary['jev_decisions']} "
                f"api_reduction={float(summary['api_reduction_rate']) * 100:.1f}%"
            )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
