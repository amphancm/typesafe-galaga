from __future__ import annotations

from pathlib import Path
from typing import Any

from typesafe_galaga.actions import Action
from typesafe_galaga.cli import build_parser
from typesafe_galaga.policy import Decision
from typesafe_galaga.rl.play import play_rl
from typesafe_galaga.state import GalagaSnapshot


def test_cli_rl_play_defaults_to_galaga() -> None:
    parser = build_parser()
    args = parser.parse_args(["rl-play"])
    assert args.rom == "roms/Galaga (U).nes"
    assert args.model == Path("models/galaga_ppo.zip")
    assert args.device == "cpu"

    for subcmd in ("rl-bc", "rl-train", "rl-eval"):
        assert parser.parse_args([subcmd]).device == "cpu"


def test_play_rl_headless_with_stubbed_env_and_policy(monkeypatch: Any, tmp_path: Path) -> None:
    snap = GalagaSnapshot(
        goal="Survive",
        stage=1,
        score=200,
        lives=3,
        x=100,
        y=199,
        dx=0,
        alive=True,
        dead=False,
        clear=False,
        active_player_bullets=0,
    )

    class _FakeRLPolicy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def choose(self, s: GalagaSnapshot, actions: Any) -> Decision:
            del s, actions
            return Decision(
                action=Action.FIRE, confidence=0.95, probabilities={"fire": 0.95}, latency_ms=0.2
            )

    class _FakeEnv:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.last_snapshot = snap
            self.last_frame = None
            self.last_step_frames = ()
            self._steps = 0

        def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
            del seed
            return None, {"score": 0, "lives": 3}

        def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            del action
            self._steps += 1
            done = self._steps >= 3
            return None, 1.0, done, False, {"score": 200, "lives": 3, "dead": done, "clear": False}

        def close(self) -> None:
            pass

    monkeypatch.setattr("typesafe_galaga.rl.play.RLPolicy", _FakeRLPolicy)
    monkeypatch.setattr("typesafe_galaga.rl.play.GalagaRLEnv", _FakeEnv)

    summary = play_rl(tmp_path / "dummy.zip", display="none", sound=False, max_steps=10)
    assert summary["score"] == 200
    assert summary["steps"] == 3
