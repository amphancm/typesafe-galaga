# Audio-Aware Reinforcement Learning (Sound Event Flags)

This document explains how `typesafe-galaga` feeds game audio signals into PPO and
Behavior Cloning policies without rendering raw audio waveforms. For the full PPO v2
training pipeline (6-action space, 4-step frame stacking, reward shaping, and asymmetric
actor-critic architecture), see [docs/ppo_v2.md](ppo_v2.md).

## Why Sound Event Flags Instead of Waveforms?

When training an RL agent against a game whose state or memory we control, rendering
44.1 kHz PCM audio and running a spectrogram/CNN encoder to decode it back into events
adds heavy CPU overhead and sample inefficiency. Instead, `typesafe-galaga` extracts a
compact **10-dimensional binary semantic audio event vector** directly from Galaga's live
NES RAM sound engine (`$0580-$06E7`) and appends it to the observation vector:

- **Zero audio rendering overhead** during headless training (no `libgme` or
  `pygame.mixer` required).
- **Frame-accurate** across multi-frame macro steps (`frames_per_decision`, default `4`).
- **Portable to custom games**: any custom adapter can populate
  `GalagaSnapshot.audio_events` directly with event names.

## Data Flow

```text
NES RAM ($0580-$06E7)
  ├─ RomSoundTracker.sync(ram)    -> newly started/re-triggered slots per frame
  └─ active_rom_sound_slots(ram)  -> currently playing slots (status != 0) per frame
          │
          ▼ (union across all frames in the RL decision step)
  slots_to_audio_events(...)      -> tuple[str, ...] canonical event names
          │
          ▼
  GalagaSnapshot.audio_events     -> also serialized in to_state()["audio"]["events"]
          │
          ▼
  encode_observation(snapshot, include_audio=True)
          │
          ▼
  float32 observation vector (166 dims = 156 base + 10 audio flags)
          │
          ▼ (optional VecFrameStack / FrameStackBuffer, default 4 steps in PPO v2)
  664-dim stacked observation (166 × 4)
```

If an environment uses a custom/mock adapter without NES RAM (`env.ram is None`) and does
not populate `snapshot.audio_events`, `GalagaRLEnv.step()` automatically falls back to
`detect_audio_events(previous_snapshot, current_snapshot)`.

## Semantic Audio Event Vector (`10` Dimensions)

In the current 6-action codebase (PPO v2), indices `0..155` (`BASE_OBSERVATION_DIM = 156`,
up from `154` in the 4-action v1 layout due to the 2 added move-and-shoot actions in the
previous-action one-hot) hold the state and radar features. The audio event flags occupy
indices `156..165` (`AUDIO_EVENT_DIM = 10`, single-step `OBSERVATION_DIM = 166`), each
set to `1.0` if triggered or active during the step and `0.0` otherwise:

| Obs Index | Flag Name | Galaga NES ROM Slot(s) | Meaning |
| :---: | --- | :---: | --- |
| `156` | `laser_shot` | `34` | Player fired a laser shot |
| `157` | `enemy_hit` | `30` | Standard enemy hit / destroyed |
| `158` | `boss_hit` | `28` | Boss Galaga damaged (first hit) |
| `159` | `boss_destroy` | `29` | Boss Galaga destroyed |
| `160` | `dive_swoop` | `32` | Enemy diving swoop sound cue |
| `161` | `ship_explode` | `35` | Player ship explosion / death |
| `162` | `one_up` | `7, 8` | Extra life jingle |
| `163` | `stage_clear` | `13..18` | Stage clear / challenging stage fanfare |
| `164` | `stage_start` | `3..6` | Stage start theme |
| `165` | `ambient` | `36, 37` | Background formation ambient hum |

In PPO v2, the `stage_clear` audio flag also participates in reward shaping: a rising
edge on `stage_clear` (gated by a 50-step cooldown) awards `stage_clear_bonus` (`+25.0`)
even when RAM stage counters lag behind the fanfare.

Defined in:
- `AUDIO_EVENT_NAMES` & `AUDIO_EVENT_SLOTS` in `src/typesafe_galaga/audio.py`
- `AUDIO_EVENT_DIM`, `AUDIO_EVENT_INDEX`, `encode_audio_events`, `encode_observation`,
  and `FrameStackBuffer` in `src/typesafe_galaga/rl/observations.py`

## Triggered + Active Semantics

Each RL step advances the emulator by `frames_per_decision` frames (default `4`). Within
`GalagaEmulatorAdapter.step()`:

1. `RomSoundTracker.sync(ram)` catches short transient sounds (such as `laser_shot` or
   `enemy_hit`) that start or re-trigger on any frame inside the 4-frame window.
2. `active_rom_sound_slots(ram)` catches ongoing sounds (such as `dive_swoop` or
   `ambient`) whose slot status byte at `$0580 + slot` is non-zero.
3. The union of triggered and active slots across all frames in the step is mapped to
   `AUDIO_EVENT_NAMES`.

## Configuration & Auto-Detection

- **Default:** `include_audio_obs=True` (`166`-dim single step) in `GalagaRLEnv`,
  `TrainConfig`, `rl-train`, and `rl-bc`. `rl-train` also defaults to `--frame-stack 4`
  (`664`-dim stacked input).
- **Opt-out:** pass `--no-audio-obs` to `rl-train` or `rl-bc` for the non-audio
  observation space (`156`-dim single step, or `624`-dim with `--frame-stack 4`).
- **Auto-detection:** `rl-eval` and `rl-play` (via `infer_obs_config_from_shape()`)
  inspect `model.observation_space.shape` when loading a `.zip` checkpoint:
  - Multiple of `166` (e.g. `(166,)`, `(664,)`) → `include_audio_obs=True`,
    `frame_stack = dim // 166`
  - Multiple of `156` (e.g. `(156,)`, `(624,)`) → `include_audio_obs=False`,
    `frame_stack = dim // 156`
- **Resuming training:** if `--resume-from` points to a checkpoint whose observation space
  does not match `--audio-obs`/`--no-audio-obs` and `--frame-stack`, `train_ppo()` raises
  a `ValueError` prompting you to match those flags or train from scratch.

## Adapting to Your Own Custom Game

To train an RL agent with sound flags on a custom game environment:

1. Emit semantic sound event strings from your game loop on each step (e.g.,
   `("laser_shot", "dive_swoop")`).
2. Pass them into `GalagaSnapshot(..., audio_events=events)` (or your equivalent state
   dataclass).
3. If your game has different sound categories, update `AUDIO_EVENT_NAMES` in
   `src/typesafe_galaga/audio.py` — `AUDIO_EVENT_DIM`, `AUDIO_EVENT_INDEX`, and
   `OBSERVATION_DIM` update automatically.

## Benchmark Results

Evaluated across 20 deterministic episodes (`seed=1000`, `frames_per_decision=4`,
`random_noop_max=4`):

| Model | Actions | Obs Dim | Steps | Mean Score | Min Score | Max Score | Mean Reward (± std) | Mean Steps |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `models/galaga_ppo_best.zip` (v1 legacy) | 4 | 154 | 1.5M | `17,624.0` | `10,160` | `21,600` | `156.16 ± 48.03` | `1,729.5` |
| `models/galaga_ppo_audio_best.zip` (v1) | 4 | 164 | 1.5M | `18,698.0` (+6.1%) | `10,290` | `24,530` | `164.77 ± 67.15` | `1,729.0` |
| `models/galaga_ppo_audio.zip` (v1) | 4 | 164 | 1.5M | `18,945.5` (+7.5%) | `10,490` | `25,000` | `169.83 ± 65.19` | `1,971.9` |
| `models/galaga_ppo_v2_best.zip` (v2) | 6 | 664 (`166×4`) | 4.6M | `25,053.5` (+42.2%) | `10,250` | `37,770` | `223.28 ± 89.96` | `2,378.0` |
| `models/galaga_ppo_v2.zip` (v2) | 6 | 664 (`166×4`) | 5.0M | **`31,269.5`** (+77.4%) | **`20,930`** | **`51,370`** | **`287.57 ± 141.78`** | **`2,831.0`** |

### Reproducing PPO v2 Training & Evaluation

```bash
# Train 5M steps with audio event flags + 4-step frame stack (664-dim)
.venv/bin/python -m typesafe_galaga.rl.train \
  --rom "roms/Galaga (U).nes" \
  --timesteps 5000000 \
  --num-envs 8 \
  --model-path models/galaga_ppo_v2.zip \
  --checkpoint-dir models/checkpoints_v2 \
  --tensorboard-log runs/galaga_ppo_v2 \
  --device cuda

# Evaluate PPO v2 model (auto-detects audio flags and frame stack)
.venv/bin/python -m typesafe_galaga.rl.evaluate \
  --model models/galaga_ppo_v2.zip \
  --rom "roms/Galaga (U).nes" \
  --episodes 20 \
  --seed 1000 \
  --output artifacts/eval_galaga_ppo_v2.json

# Watch the agent play with live dashboard + NES ROM audio
.venv/bin/python -m typesafe_galaga.cli rl-play \
  --model models/galaga_ppo_v2.zip \
  --rom "roms/Galaga (U).nes"
```
