from __future__ import annotations

from types import SimpleNamespace

from typesafe_galaga.runner import GalagaEnvWrapper
from typesafe_galaga.state import LIVES_RESERVE_ADDR, OAM_START, PLAYER_Y, TILE_PLAYER_SHIP


def _make_wrapper(ram: bytearray) -> GalagaEnvWrapper:
    wrapper = object.__new__(GalagaEnvWrapper)
    wrapper._nes_env = SimpleNamespace(ram=ram)  # type: ignore[assignment]
    wrapper._lives = 3
    wrapper._absent_frames = 0
    wrapper._was_alive = True
    wrapper._last_score = 0
    wrapper._stage = 1
    wrapper._has_start_backup = False
    return wrapper


def _set_ship(ram: bytearray, present: bool) -> None:
    if present:
        ram[OAM_START] = PLAYER_Y
        ram[OAM_START + 1] = TILE_PLAYER_SHIP
    else:
        ram[OAM_START] = 0xFF
        ram[OAM_START + 1] = 0xFF


def test_enrich_info_reads_30_lives_from_ram_and_handles_last_respawn() -> None:
    ram = bytearray(0x0800)
    ram[LIVES_RESERVE_ADDR] = 29
    _set_ship(ram, True)
    wrapper = _make_wrapper(ram)

    info, dead = wrapper._enrich_info({})
    assert info["lives"] == 30
    assert dead is False

    # Die 1st time: reserve is still 29 until respawn.
    _set_ship(ram, False)
    for _ in range(20):
        info, dead = wrapper._enrich_info({})
    assert info["lives"] == 29
    assert dead is False

    # Respawn with 28 reserve -> 29 total lives.
    ram[LIVES_RESERVE_ADDR] = 28
    _set_ship(ram, True)
    info, dead = wrapper._enrich_info({})
    assert info["lives"] == 29
    assert dead is False

    # Simulate final respawn (reserve drops 1 -> 0 before ship sprite reappears).
    ram[LIVES_RESERVE_ADDR] = 1
    _set_ship(ram, False)
    for _ in range(20):
        info, dead = wrapper._enrich_info({})
    assert info["lives"] == 1
    assert dead is False

    ram[LIVES_RESERVE_ADDR] = 0
    info, dead = wrapper._enrich_info({})
    assert info["lives"] == 1
    assert dead is False

    _set_ship(ram, True)
    info, dead = wrapper._enrich_info({})
    assert info["lives"] == 1
    assert dead is False

    # Final death with 0 reserve -> Game Over.
    _set_ship(ram, False)
    for _ in range(20):
        info, dead = wrapper._enrich_info({})
    assert info["lives"] == 0
    assert dead is True
