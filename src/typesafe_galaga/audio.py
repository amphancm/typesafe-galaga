from __future__ import annotations

import ctypes
import ctypes.util
import struct
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .state import GalagaSnapshot

SAMPLE_RATE = 44_100
DEFAULT_ROM_PATH = "roms/Galaga (U).nes"

# Galaga (U).nes 6502 sound driver constants (verified from PRG-ROM disassembly)
_INES_MAGIC = b"NES\x1a"
_INES_HEADER_SIZE = 16
_PRG_ROM_SIZE = 32_768
_NSF_LOAD_ADDR = 0x8000
_GALAGA_SOUND_INIT_ADDR = 0xEADA
_GALAGA_SOUND_PLAY_ADDR = 0xEB08
_GALAGA_GAME_ACTIVE_ZP = 0x62
_GALAGA_PAUSED_ZP = 0x5B

# Live NES RAM layout for Galaga's 39-slot sound engine ($0580-$06E7)
ROM_SOUND_SLOT_COUNT = 39
ROM_SOUND_STATUS_ADDR = 0x0580
ROM_SOUND_CHANNEL_BASE = 0x05B0
ROM_SOUND_CHANNEL_STRIDE = 8
ROM_SOUND_SEQ_OFFSET = 5

# Verified Galaga (U).nes sound slot mappings
ROM_BGM_SLOTS: dict[str, tuple[int, ...]] = {
    "stage_start": (3, 4, 5, 6),
    "overworld": (3, 4, 5, 6),
    "galaga": (3, 4, 5, 6),
    "challenging_stage": (13, 14, 15, 16, 17, 18),
    "ambient": (36, 37),
}

ROM_SFX_SLOTS: dict[str, tuple[int, ...]] = {
    "laser_shot": (34,),
    "enemy_hit": (30,),
    "boss_hit": (28,),
    "boss_destroy": (29,),
    "dive_swoop": (32,),
    "ship_explode": (35,),
    "death": (35,),
    "one_up": (7, 8),
    "stage_clear": (13, 14, 15, 16, 17, 18),
}

# Slots whose ROM sound scripts loop continuously until cleared in RAM
_LOOPING_ROM_SLOTS: frozenset[int] = frozenset({24, 27, 33, 38})

_GME_LIB_CACHE: ctypes.CDLL | None = None
_GME_LOAD_ATTEMPTED = False


def _load_libgme() -> ctypes.CDLL | None:
    """Load libgme (Game Music Emu) for authentic NES 2A03 APU NSF emulation."""
    global _GME_LIB_CACHE, _GME_LOAD_ATTEMPTED
    if _GME_LOAD_ATTEMPTED:
        return _GME_LIB_CACHE
    _GME_LOAD_ATTEMPTED = True

    candidates: list[str] = ["libgme.so.0", "libgme.so", "libgme.dylib", "gme.dll"]
    found = ctypes.util.find_library("gme")
    if found and found not in candidates:
        candidates.insert(0, found)

    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            lib.gme_open_data.argtypes = [
                ctypes.c_void_p,
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
            ]
            lib.gme_open_data.restype = ctypes.c_char_p
            lib.gme_start_track.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.gme_start_track.restype = ctypes.c_char_p
            lib.gme_play.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int16),
            ]
            lib.gme_play.restype = ctypes.c_char_p
            lib.gme_delete.argtypes = [ctypes.c_void_p]
            lib.gme_delete.restype = None
            lib.gme_ignore_silence.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.gme_ignore_silence.restype = None
            _GME_LIB_CACHE = lib
            return lib
        except OSError:
            continue
    return None


def _load_galaga_prg_rom(rom_path: str | Path) -> bytes:
    """Load and validate the 32KB PRG-ROM bank from a Galaga (U).nes file."""
    path = Path(rom_path)
    if not path.exists():
        raise FileNotFoundError(f"Galaga ROM not found at: {path}")
    data = path.read_bytes()
    if len(data) < _INES_HEADER_SIZE + _PRG_ROM_SIZE or data[:4] != _INES_MAGIC:
        raise ValueError(f"Invalid NES ROM file: {path}")
    return data[_INES_HEADER_SIZE : _INES_HEADER_SIZE + _PRG_ROM_SIZE]


def _build_nsf_for_slots(prg_rom: bytes, slots: Sequence[int]) -> bytes:
    """Wrap Galaga's PRG-ROM into an in-memory NSF image that triggers the given slots."""
    if not slots:
        raise ValueError("At least one ROM sound slot must be specified")
    for slot in slots:
        if not 0 <= int(slot) < ROM_SOUND_SLOT_COUNT:
            raise ValueError(
                f"ROM sound slot must be in 0..{ROM_SOUND_SLOT_COUNT - 1}, got {slot}"
            )

    # 6502 INIT routine placed in Galaga's free $FF padding at $8000:
    #   JSR $EADA        ; Galaga sound init
    #   LDA #$01 / STA $62 ; game active flag
    #   LDA #$00 / STA $5B ; unpaused flag
    #   LDA #$01
    #   STA $0580 + slot ; trigger each requested sound slot
    #   RTS
    stub = bytearray(
        [
            0x20,
            _GALAGA_SOUND_INIT_ADDR & 0xFF,
            (_GALAGA_SOUND_INIT_ADDR >> 8) & 0xFF,
            0xA9,
            0x01,
            0x85,
            _GALAGA_GAME_ACTIVE_ZP,
            0xA9,
            0x00,
            0x85,
            _GALAGA_PAUSED_ZP,
            0xA9,
            0x01,
        ]
    )
    for slot in slots:
        addr = ROM_SOUND_STATUS_ADDR + int(slot)
        stub.extend((0x8D, addr & 0xFF, (addr >> 8) & 0xFF))
    stub.append(0x60)  # RTS

    prg = bytearray(prg_rom)
    prg[: len(stub)] = stub

    header = bytearray(128)
    header[0:5] = b"NESM\x1a"
    header[5] = 1  # NSF version 1
    header[6] = 1  # 1 track
    header[7] = 1  # starting track 1
    struct.pack_into(
        "<HHH",
        header,
        8,
        _NSF_LOAD_ADDR,
        _NSF_LOAD_ADDR,
        _GALAGA_SOUND_PLAY_ADDR,
    )
    header[14:20] = b"Galaga"
    struct.pack_into("<H", header, 0x6E, 16_666)  # NTSC 60 Hz playback rate
    header[0x7A] = 0  # NTSC
    return bytes(header) + bytes(prg)


@dataclass(frozen=True, slots=True)
class RenderedRomAudio:
    pcm: np.ndarray
    looping: bool


class GalagaRomAudioBank:
    """Renders and caches authentic NES 2A03 PCM audio directly from Galaga (U).nes."""

    def __init__(
        self,
        rom_path: str | Path = DEFAULT_ROM_PATH,
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.rom_path = Path(rom_path)
        self.sample_rate = int(sample_rate)
        self._prg_rom = _load_galaga_prg_rom(self.rom_path)
        self._libgme = _load_libgme()
        if self._libgme is None:
            raise RuntimeError(
                "libgme is required to render NES ROM audio (e.g. libgme.so.0 not found)"
            )
        self._cache: dict[tuple[int, ...], RenderedRomAudio] = {}

    def render_slots(
        self,
        slots: Sequence[int],
        *,
        max_duration_s: float = 8.0,
        gain: float = 1.6,
    ) -> RenderedRomAudio:
        """Execute Galaga's 6502 sound driver on the given slots and return stereo int16 PCM."""
        key = tuple(int(s) for s in slots)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        assert self._libgme is not None
        nsf_bytes = _build_nsf_for_slots(self._prg_rom, key)

        emu = ctypes.c_void_p()
        err = self._libgme.gme_open_data(
            nsf_bytes,
            len(nsf_bytes),
            ctypes.byref(emu),
            self.sample_rate,
        )
        if err:
            raise RuntimeError(f"libgme failed to open Galaga NSF data: {err.decode('utf-8', 'replace')}")

        try:
            self._libgme.gme_ignore_silence(emu, 1)
            err = self._libgme.gme_start_track(emu, 0)
            if err:
                raise RuntimeError(
                    f"libgme failed to start Galaga track: {err.decode('utf-8', 'replace')}"
                )

            chunk_frames = max(256, self.sample_rate // 20)  # 50 ms chunks
            chunk_samples = chunk_frames * 2
            buf = (ctypes.c_int16 * chunk_samples)()
            max_chunks = max(1, round(max_duration_s * 20))

            chunks: list[np.ndarray] = []
            started = False
            silent_chunks = 0
            last_active_chunk = -1

            for idx in range(max_chunks):
                err = self._libgme.gme_play(emu, chunk_samples, buf)
                if err:
                    raise RuntimeError(
                        f"libgme playback error: {err.decode('utf-8', 'replace')}"
                    )
                frame_chunk = np.ctypeslib.as_array(buf).reshape(-1, 2).copy()
                chunks.append(frame_chunk)
                peak = int(np.max(np.abs(frame_chunk)))
                if peak > 32:
                    started = True
                    silent_chunks = 0
                    last_active_chunk = idx
                elif started:
                    silent_chunks += 1
                    if silent_chunks >= 4:  # 200 ms of silence -> sound finished
                        break

            looping = (
                len(key) == 1 and key[0] in _LOOPING_ROM_SLOTS
            ) or (started and silent_chunks == 0 and len(chunks) == max_chunks)

            if last_active_chunk >= 0:
                end_chunk = min(len(chunks), last_active_chunk + 2)
                pcm = np.concatenate(chunks[:end_chunk], axis=0)
            else:
                pcm = np.zeros((chunk_frames, 2), dtype=np.int16)

            if gain != 1.0 and pcm.size > 0:
                scaled = np.clip(pcm.astype(np.float32) * float(gain), -32767.0, 32767.0)
                pcm = scaled.astype(np.int16)

            rendered = RenderedRomAudio(pcm=np.ascontiguousarray(pcm), looping=looping)
            self._cache[key] = rendered
            return rendered
        finally:
            self._libgme.gme_delete(emu)


_DEFAULT_BANK_CACHE: dict[tuple[str, int], GalagaRomAudioBank] = {}


def _get_audio_bank(
    rom_path: str | Path = DEFAULT_ROM_PATH,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> GalagaRomAudioBank:
    key = (str(Path(rom_path).resolve()), int(sample_rate))
    bank = _DEFAULT_BANK_CACHE.get(key)
    if bank is None:
        bank = GalagaRomAudioBank(rom_path=rom_path, sample_rate=sample_rate)
        _DEFAULT_BANK_CACHE[key] = bank
    return bank


def render_rom_slots(
    slots: Sequence[int],
    *,
    rom_path: str | Path = DEFAULT_ROM_PATH,
    sample_rate: int = SAMPLE_RATE,
    max_duration_s: float = 8.0,
) -> np.ndarray:
    """Render one or more Galaga ROM sound slots as stereo int16 PCM."""
    bank = _get_audio_bank(rom_path, sample_rate=sample_rate)
    return bank.render_slots(slots, max_duration_s=max_duration_s).pcm


def render_rom_slot(
    slot: int,
    *,
    rom_path: str | Path = DEFAULT_ROM_PATH,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render a single Galaga ROM sound slot (0..38) as stereo int16 PCM."""
    return render_rom_slots((slot,), rom_path=rom_path, sample_rate=sample_rate)


def extract_rom_bgm(
    theme: str = "stage_start",
    *,
    rom_path: str | Path = DEFAULT_ROM_PATH,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render a Galaga BGM/jingle directly from the NES ROM as stereo int16 PCM."""
    slots = ROM_BGM_SLOTS.get(theme, ROM_BGM_SLOTS["stage_start"])
    return render_rom_slots(slots, rom_path=rom_path, sample_rate=sample_rate, max_duration_s=8.0)


def extract_rom_sfx(
    name: str,
    *,
    rom_path: str | Path = DEFAULT_ROM_PATH,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render a Galaga SFX directly from the NES ROM as stereo int16 PCM."""
    slots = ROM_SFX_SLOTS.get(name)
    if slots is None:
        raise ValueError(f"Unknown SFX name: {name!r}")
    return render_rom_slots(slots, rom_path=rom_path, sample_rate=sample_rate, max_duration_s=4.0)


class RomSoundTracker:
    """Tracks Galaga's 39 sound-engine slots directly in live NES RAM."""

    def __init__(self) -> None:
        self._prev_status: list[int] = [0] * ROM_SOUND_SLOT_COUNT
        self._prev_seq: list[int] = [0] * ROM_SOUND_SLOT_COUNT

    def reset(self) -> None:
        self._prev_status = [0] * ROM_SOUND_SLOT_COUNT
        self._prev_seq = [0] * ROM_SOUND_SLOT_COUNT

    def sync(self, ram: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Inspect NES RAM and return (started_slots, stopped_slots) for this frame."""
        if ram is None or len(ram) < ROM_SOUND_CHANNEL_BASE + ROM_SOUND_SLOT_COUNT * ROM_SOUND_CHANNEL_STRIDE:
            return ((), ())

        started: list[int] = []
        stopped: list[int] = []
        for slot in range(ROM_SOUND_SLOT_COUNT):
            st = int(ram[ROM_SOUND_STATUS_ADDR + slot])
            seq = int(
                ram[
                    ROM_SOUND_CHANNEL_BASE
                    + slot * ROM_SOUND_CHANNEL_STRIDE
                    + ROM_SOUND_SEQ_OFFSET
                ]
            )
            prev_st = self._prev_status[slot]
            prev_seq = self._prev_seq[slot]

            if st != 0:
                if st == 1 or prev_st == 0 or (prev_st == 2 and st == 2 and seq < prev_seq):
                    started.append(slot)
            elif prev_st != 0:
                stopped.append(slot)

            self._prev_status[slot] = st
            self._prev_seq[slot] = seq

        return (tuple(started), tuple(stopped))


def detect_audio_events(
    prev: GalagaSnapshot | None,
    curr: GalagaSnapshot,
    *,
    finishing: bool = False,
) -> list[str]:
    """Fallback snapshot-based event detection when live NES RAM is unavailable."""
    if prev is None:
        return []

    events: list[str] = []
    if curr.dead and not prev.dead:
        return ["ship_explode"]
    if curr.clear and not prev.clear:
        return ["stage_clear"]

    if curr.lives > prev.lives:
        events.append("one_up")
    if curr.active_player_bullets > prev.active_player_bullets:
        events.append("laser_shot")

    score_delta = curr.score - prev.score
    if score_delta > 0 and not curr.clear and not finishing:
        events.append("enemy_hit")

    return events


@dataclass
class AudioConfig:
    enabled: bool = True
    volume: float = 0.5
    bgm_volume_ratio: float = 0.65
    sfx_volume_ratio: float = 0.95
    rom_path: str = DEFAULT_ROM_PATH


class GalagaAudioEngine:
    """Real-time NES ROM audio engine driven directly by Galaga (U).nes."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        if not 0.0 <= self.config.volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")

        self._muted: bool = not self.config.enabled
        self._available: bool = False
        self._mixer: Any = None
        self._pygame: Any = None
        self._bank: GalagaRomAudioBank | None = None
        self._bgm_channel: Any = None
        self._sfx_channels: list[Any] = []
        self._slot_sounds: dict[int, tuple[Any, bool]] = {}
        self._slot_channels: dict[int, Any] = {}
        self._named_sounds: dict[str, Any] = {}
        self._stage_start_sound: Any = None
        self._tracker = RomSoundTracker()
        self._prev_snapshot: GalagaSnapshot | None = None
        self._played_intro: bool = False

        if self.config.enabled:
            self._init_mixer()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def sound_on(self) -> bool:
        return self._available and not self._muted

    def _init_mixer(self) -> None:
        try:
            import pygame
        except ImportError:
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            init_info = pygame.mixer.get_init()
            if not init_info:
                return
            sample_rate = int(init_info[0])
            self._bank = _get_audio_bank(self.config.rom_path, sample_rate=sample_rate)
            pygame.mixer.set_num_channels(16)
            self._pygame = pygame
            self._mixer = pygame.mixer
            self._bgm_channel = pygame.mixer.Channel(0)
            self._sfx_channels = [pygame.mixer.Channel(i) for i in range(1, 16)]

            # Pre-render the stage start fanfare and common gameplay slots from the ROM
            intro = self._bank.render_slots(ROM_BGM_SLOTS["stage_start"], max_duration_s=8.0)
            self._stage_start_sound = pygame.sndarray.make_sound(intro.pcm)
            for slot in (28, 29, 30, 31, 32, 34, 35, 36, 37):
                self._get_slot_sound(slot)

            self._available = True
            self._apply_volumes()
        except (pygame.error, RuntimeError, OSError, ValueError):
            # Graceful fallback in headless/CI environments without an audio device or libgme.
            self._available = False
            self._mixer = None
            self._pygame = None
            self._bgm_channel = None

    def _get_slot_sound(self, slot: int) -> tuple[Any, bool] | None:
        if self._bank is None or self._pygame is None:
            return None
        cached = self._slot_sounds.get(slot)
        if cached is not None:
            return cached
        try:
            rendered = self._bank.render_slots((slot,), max_duration_s=8.0)
            sound = self._pygame.sndarray.make_sound(rendered.pcm)
            vol = 0.0 if self._muted else self.config.volume * self.config.sfx_volume_ratio
            sound.set_volume(vol)
            entry = (sound, rendered.looping)
            self._slot_sounds[slot] = entry
            return entry
        except (RuntimeError, OSError, ValueError):
            return None

    def _get_named_sfx_sound(self, name: str) -> Any | None:
        if self._bank is None or self._pygame is None:
            return None
        cached = self._named_sounds.get(name)
        if cached is not None:
            return cached
        slots = ROM_SFX_SLOTS.get(name)
        if slots is None:
            return None
        try:
            rendered = self._bank.render_slots(slots, max_duration_s=4.0)
            sound = self._pygame.sndarray.make_sound(rendered.pcm)
            vol = 0.0 if self._muted else self.config.volume * self.config.sfx_volume_ratio
            sound.set_volume(vol)
            self._named_sounds[name] = sound
            return sound
        except (RuntimeError, OSError, ValueError):
            return None

    def _apply_volumes(self) -> None:
        if not self._available or self._bgm_channel is None:
            return
        if self._muted:
            self._bgm_channel.set_volume(0.0)
            for sound, _ in self._slot_sounds.values():
                sound.set_volume(0.0)
            for sound in self._named_sounds.values():
                sound.set_volume(0.0)
            return
        self._bgm_channel.set_volume(self.config.volume * self.config.bgm_volume_ratio)
        sfx_vol = self.config.volume * self.config.sfx_volume_ratio
        for sound, _ in self._slot_sounds.values():
            sound.set_volume(sfx_vol)
        for sound in self._named_sounds.values():
            sound.set_volume(sfx_vol)

    def set_muted(self, muted: bool) -> bool:
        self._muted = bool(muted)
        self._apply_volumes()
        return self._muted

    def toggle_mute(self) -> bool:
        return self.set_muted(not self._muted)

    def _play_stage_start(self) -> None:
        if not self._available or self._muted or self._bgm_channel is None:
            return
        if self._stage_start_sound is not None:
            self._bgm_channel.play(self._stage_start_sound, loops=0)
            self._apply_volumes()

    def _alloc_channel_for_slot(self, slot: int) -> Any | None:
        existing = self._slot_channels.get(slot)
        if existing is not None:
            return existing
        for channel in self._sfx_channels:
            if not channel.get_busy():
                self._slot_channels[slot] = channel
                return channel
        if self._sfx_channels:
            channel = self._sfx_channels[slot % len(self._sfx_channels)]
            self._slot_channels[slot] = channel
            return channel
        return None

    def _play_slot(self, slot: int) -> None:
        if not self._available or self._muted:
            return
        entry = self._get_slot_sound(slot)
        if entry is None:
            return
        sound, looping = entry
        channel = self._alloc_channel_for_slot(slot)
        if channel is not None:
            channel.play(sound, loops=-1 if looping else 0)

    def _stop_slot(self, slot: int) -> None:
        if not self._available:
            return
        channel = self._slot_channels.get(slot)
        entry = self._slot_sounds.get(slot)
        if channel is not None and entry is not None:
            _, looping = entry
            if looping:
                channel.stop()

    def _play_named_sfx(self, name: str) -> None:
        if not self._available or self._muted:
            return
        sound = self._get_named_sfx_sound(name)
        if sound is None:
            return
        for channel in self._sfx_channels:
            if not channel.get_busy():
                channel.play(sound)
                return
        if self._sfx_channels:
            self._sfx_channels[0].play(sound)

    def apply_rom_events(
        self,
        started: Sequence[int],
        stopped: Sequence[int] = (),
    ) -> list[str]:
        """Play/stop authentic Galaga ROM sound slots reported by NES RAM."""
        for slot in stopped:
            self._stop_slot(int(slot))
        triggered: list[str] = []
        for slot in started:
            slot_id = int(slot)
            self._play_slot(slot_id)
            triggered.append(f"rom_slot_{slot_id:02d}")
        return triggered

    def sync_ram(self, ram: Any) -> list[str]:
        """Poll Galaga's live NES RAM sound table and play newly triggered ROM sounds."""
        started, stopped = self._tracker.sync(ram)
        return self.apply_rom_events(started, stopped)

    def reset(self) -> None:
        """Reset episode audio state (e.g. on game start or restart)."""
        if self._available:
            if self._bgm_channel is not None:
                self._bgm_channel.stop()
            for channel in self._sfx_channels:
                channel.stop()
        self._slot_channels.clear()
        self._tracker.reset()
        self._prev_snapshot = None
        self._played_intro = False

    def update(
        self,
        snapshot: GalagaSnapshot | None,
        *,
        ram: Any = None,
        area_type: int | None = None,
        finishing: bool = False,
    ) -> list[str]:
        """Advance audio state from live NES RAM (preferred) or snapshot fallback."""
        del area_type
        if not isinstance(snapshot, GalagaSnapshot):
            return []

        if not self._played_intro and not snapshot.dead and not snapshot.clear:
            self._played_intro = True
            self._play_stage_start()

        if ram is not None:
            triggered = self.sync_ram(ram)
            self._prev_snapshot = snapshot
            return triggered

        events = detect_audio_events(self._prev_snapshot, snapshot, finishing=finishing)
        for event in events:
            self._play_named_sfx(event)

        self._prev_snapshot = snapshot
        return events

    def close(self) -> None:
        self.reset()
        if self._available and self._mixer is not None:
            with suppress(RuntimeError, OSError):
                self._mixer.quit()
        self._available = False
        self._mixer = None
        self._pygame = None
        self._bgm_channel = None
        self._sfx_channels.clear()
        self._slot_channels.clear()
        self._slot_sounds.clear()
        self._named_sounds.clear()
        self._stage_start_sound = None
