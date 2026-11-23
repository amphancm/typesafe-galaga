from __future__ import annotations

from typesafe_galaga.cli import _demo_ram
from typesafe_galaga.state import GalagaStateParser


def test_parse_demo_ram_extracts_galaga_entities_and_tactical_state() -> None:
    parser = GalagaStateParser()
    snap = parser.parse({"stage": 1, "lives": 3}, _demo_ram(), previous_action="fire")

    assert snap.x == 99
    assert snap.y == 199
    assert snap.alive is True
    assert snap.score == 260
    assert snap.lives == 3
    assert snap.active_player_bullets == 1
    assert len(snap.enemies) == 2
    assert len(snap.projectiles) == 1
    assert snap.danger_imminent is True
    assert snap.recommended_evade_dir == "left"
    assert snap.aligned_enemy_count == 1
    assert len(snap.radar_grid) == 9
    assert len(snap.radar_grid[0]) == 11

    state = snap.to_state()
    assert state["player"]["x"] == 99
    assert state["episode"]["score"] == 260
    assert state["tactical"]["danger_imminent"] is True
    assert "Radar" in snap.to_text()


def test_parse_buffer_and_sequence_ram_produce_identical_snapshots() -> None:
    ram_bytes = _demo_ram()
    ram_list = list(ram_bytes)

    snap_buf = GalagaStateParser().parse({"stage": 1, "lives": 3}, ram_bytes)
    snap_seq = GalagaStateParser().parse({"stage": 1, "lives": 3}, ram_list)
    assert snap_buf == snap_seq
