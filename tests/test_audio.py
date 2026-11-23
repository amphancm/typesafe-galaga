from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from typesafe_galaga.actions import Action
from typesafe_galaga.audio import (
    DEFAULT_ROM_PATH,
    ROM_SOUND_CHANNEL_BASE,
    ROM_SOUND_CHANNEL_STRIDE,
    ROM_SOUND_SEQ_OFFSET,
    ROM_SOUND_STATUS_ADDR,
    AudioConfig,
    GalagaAudioEngine,
    RomSoundTracker,
    detect_audio_events,
    extract_rom_bgm,
    extract_rom_sfx,
    render_rom_slot,
    render_rom_slots,
)
from typesafe_galaga.cli import build_parser
from typesafe_galaga.dashboard import DashboardCommand, DashboardConfig
from typesafe_galaga.policy import Decision
from typesafe_galaga.rl.play import play_rl
from typesafe_galaga.state import GalagaSnapshot


def _snapshot(**overrides: Any) -> GalagaSnapshot:
    base = GalagaSnapshot(
        goal="Survive",
        stage=1,
        score=0,
        lives=2,
        x=100,
        y=199,
        dx=0,
        alive=True,
        dead=False,
        clear=False,
        active_player_bullets=0,
    )
    return replace(base, **overrides)


@pytest.mark.parametrize("theme", ["stage_start", "challenging_stage", "ambient"])
def test_extract_rom_bgm_returns_valid_stereo_int16(theme: str) -> None:
    if not Path(DEFAULT_ROM_PATH).exists():
        pytest.skip("Galaga ROM not present")
    pcm = extract_rom_bgm(theme, sample_rate=22_050)
    assert pcm.dtype == np.int16
    assert pcm.ndim == 2
    assert pcm.shape[1] == 2
    assert pcm.shape[0] > 22_050
    assert int(np.max(np.abs(pcm))) > 1000


@pytest.mark.parametrize(
    "sfx_name",
    [
        "laser_shot",
        "enemy_hit",
        "boss_hit",
        "boss_destroy",
        "dive_swoop",
        "ship_explode",
        "one_up",
        "stage_clear",
    ],
)
def test_extract_rom_sfx_returns_valid_stereo_int16(sfx_name: str) -> None:
    if not Path(DEFAULT_ROM_PATH).exists():
        pytest.skip("Galaga ROM not present")
    pcm = extract_rom_sfx(sfx_name, sample_rate=22_050)
    assert pcm.dtype == np.int16
    assert pcm.ndim == 2
    assert pcm.shape[1] == 2
    assert pcm.shape[0] > 1000
    assert int(np.max(np.abs(pcm))) > 1000


def test_render_rom_slot_and_validation() -> None:
    if not Path(DEFAULT_ROM_PATH).exists():
        pytest.skip("Galaga ROM not present")
    single = render_rom_slot(34, sample_rate=22_050)
    assert single.dtype == np.int16
    assert single.ndim == 2 and single.shape[1] == 2
    assert int(np.max(np.abs(single))) > 1000

    multi = render_rom_slots((3, 4, 5, 6), sample_rate=22_050)
    assert multi.dtype == np.int16
    assert multi.shape[0] > 22_050 * 4

    with pytest.raises(ValueError, match="ROM sound slot"):
        render_rom_slot(99, sample_rate=22_050)
    with pytest.raises(ValueError, match="Unknown SFX name"):
        extract_rom_sfx("nonexistent_sfx", sample_rate=22_050)


def test_rom_sound_tracker_detects_starts_retriggers_and_stops() -> None:
    tracker = RomSoundTracker()
    ram = bytearray(0x0800)

    assert tracker.sync(ram) == ((), ())

    # Trigger laser shot (slot 34) with status 1
    ram[ROM_SOUND_STATUS_ADDR + 34] = 1
    assert tracker.sync(ram) == ((34,), ())

    # Next frame: driver transitions 1 -> 2 (must NOT double-trigger)
    ram[ROM_SOUND_STATUS_ADDR + 34] = 2
    ram[ROM_SOUND_CHANNEL_BASE + 34 * ROM_SOUND_CHANNEL_STRIDE + ROM_SOUND_SEQ_OFFSET] = 3
    assert tracker.sync(ram) == ((), ())

    # Mid-play re-trigger: status stays 2, sequence index resets to 0
    ram[ROM_SOUND_CHANNEL_BASE + 34 * ROM_SOUND_CHANNEL_STRIDE + ROM_SOUND_SEQ_OFFSET] = 0
    assert tracker.sync(ram) == ((34,), ())

    # Slot finishes: status goes 2 -> 0
    ram[ROM_SOUND_STATUS_ADDR + 34] = 0
    assert tracker.sync(ram) == ((), (34,))


def test_detect_audio_events_covers_core_gameplay_transitions() -> None:
    base = _snapshot()
    assert detect_audio_events(None, base) == []

    # Laser shot fired
    fired = _snapshot(active_player_bullets=1)
    assert detect_audio_events(base, fired) == ["laser_shot"]

    # Enemy hit (score increases)
    assert detect_audio_events(base, _snapshot(score=100)) == ["enemy_hit"]

    # 1-UP
    assert detect_audio_events(base, _snapshot(lives=3)) == ["one_up"]

    # Stage clear and ship explosion
    assert detect_audio_events(base, _snapshot(clear=True)) == ["stage_clear"]
    assert detect_audio_events(base, _snapshot(dead=True)) == ["ship_explode"]


def test_audio_engine_mute_toggle_and_disabled_mode() -> None:
    engine = GalagaAudioEngine(AudioConfig(enabled=False, volume=0.5))
    assert not engine.available
    assert engine.muted is True
    assert engine.toggle_mute() is False
    assert engine.toggle_mute() is True
    assert engine.update(_snapshot()) == []
    engine.close()

    with pytest.raises(ValueError, match="volume must be between"):
        GalagaAudioEngine(AudioConfig(enabled=False, volume=1.5))


def test_cli_sound_and_volume_flags() -> None:
    args_default = build_parser().parse_args(["rl-play"])
    assert args_default.sound is True
    assert args_default.volume == pytest.approx(0.5)

    args_muted = build_parser().parse_args(["rl-play", "--no-sound", "--volume", "0.25"])
    assert args_muted.sound is False
    assert args_muted.volume == pytest.approx(0.25)

    play_args = build_parser().parse_args(["play", "--no-sound"])
    assert play_args.sound is False


def test_play_rl_wires_audio_and_handles_mute_toggle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_events: list[str] = []

    class FakeAudio:
        def __init__(self, config: AudioConfig | None = None) -> None:
            self.config = config or AudioConfig()
            self.muted = not self.config.enabled
            audio_events.append(f"init:{self.config.enabled}:{self.config.volume}")

        def reset(self) -> None:
            audio_events.append("reset")

        def apply_rom_events(self, started: Any, stopped: Any = ()) -> list[str]:
            del stopped
            audio_events.append(f"rom:{tuple(started)}")
            return []

        def update(
            self,
            snapshot: Any,
            *,
            ram: Any = None,
            area_type: int | None = None,
            finishing: bool = False,
        ) -> list[str]:
            del snapshot, ram, area_type, finishing
            audio_events.append("update")
            return []

        def toggle_mute(self) -> bool:
            self.muted = not self.muted
            audio_events.append(f"mute:{self.muted}")
            return self.muted

        def close(self) -> None:
            audio_events.append("close")

    class FakeRLPolicy:
        def __init__(
            self, model_path: Path, *, deterministic: bool = True, device: str = "cpu"
        ) -> None:
            del model_path, deterministic, device

        def choose(self, snapshot: Any, actions: Any) -> Decision:
            del snapshot, actions
            return Decision(
                action=Action.RIGHT, confidence=1.0, probabilities={"right": 1.0}, latency_ms=1.0
            )

    class FakeEnv:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.last_snapshot = _snapshot()
            self.last_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            self._frames: tuple[Any, ...] = (self.last_frame,)

        @property
        def last_step_frames(self) -> tuple[Any, ...]:
            return self._frames

        @property
        def last_step_audio_events(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
            return (((34,), ()),)

        def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
            del seed
            return np.zeros(1), {"clear": False, "dead": False, "best_x": 40}

        def step(self, action_idx: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            del action_idx
            self.last_snapshot = _snapshot(clear=True)
            return (
                np.zeros(1),
                1.0,
                True,
                False,
                {"clear": True, "dead": False, "best_x": 100, "frames_executed": 1},
            )

        def close(self) -> None:
            return None

    draw_count = 0

    class FakeDashboard:
        def __init__(self, config: DashboardConfig | None = None) -> None:
            del config

        def draw(self, frame: Any, snapshot: Any, decision: Any, **kwargs: Any) -> DashboardCommand:
            nonlocal draw_count
            del frame, snapshot, decision
            draw_count += 1
            if draw_count == 1:
                assert kwargs.get("sound_enabled") is True
                return DashboardCommand.TOGGLE_MUTE
            assert kwargs.get("sound_enabled") is False
            return DashboardCommand.QUIT

        def close(self) -> None:
            return None

    monkeypatch.setattr("typesafe_galaga.rl.play.GalagaAudioEngine", FakeAudio)
    monkeypatch.setattr("typesafe_galaga.rl.play.RLPolicy", FakeRLPolicy)
    monkeypatch.setattr("typesafe_galaga.rl.play.GalagaRLEnv", FakeEnv)
    monkeypatch.setattr("typesafe_galaga.rl.play.LiveDashboard", FakeDashboard)

    play_rl(tmp_path / "dummy.zip", display="dashboard", sound=True, volume=0.6)
    assert "init:True:0.6" in audio_events
    assert "reset" in audio_events
    assert "rom:(34,)" in audio_events
    assert "mute:True" in audio_events
    assert audio_events[-1] == "close"
