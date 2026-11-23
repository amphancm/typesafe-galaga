# TypeSafe Galaga (`typesafe-galaga`)

An experimental controller that lets TypeSafe's Jev model—or a local reinforcement
learning (PPO) agent—choose NES controller inputs for **Galaga**.

The controllers do not receive screenshots. The harness parses emulator RAM and OAM
sprite data into structured state containing the player's position, bullets, enemies,
projectiles, tactical danger cues, score, lives, stage, and semantic sound event flags.
Jev receives this state as JSON; the RL policy receives a bounded 164-dimensional numeric
observation (154 state/radar features + 10 semantic audio event flags). Each controller
action advances the emulator before the next decision.

```text
NES emulator -> RAM/OAM + sound flags -> structured JSON -> Jev Choice -> controller input
                                      -> 164-dim vector  -> PPO Policy -> controller input
                                                         -> Hybrid (PPO + Jev escalation)
```

The controller actions are `noop`, `left`, `right`, `fire`.

## Requirements

- Python 3.13 or newer
- A legally obtained Galaga NES ROM (not distributed with the project)
- `TYPESAFE_API_KEY` for TypeSafe-controlled `play` or hybrid `rl-play`; local RL does
  not require an API key

## Setup

On Ubuntu, install the SDL development libraries required by `pygame-ce`, then create
an environment and install the Galaga, RL, and development extras:

```bash
sudo apt update
sudo apt install -y \
  python3-venv python3-dev build-essential \
  libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
  libportmidi-dev libfreetype6-dev
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[galaga,rl,dev]"
export TYPESAFE_API_KEY="your-key"
```

The emulator ROM path defaults to `roms/Galaga (U).nes`. Override it with `--rom` if
your legally obtained ROM is stored elsewhere.

## Direct control

Print a sample parsed state without launching the emulator or calling the API:

```bash
.venv/bin/typesafe-galaga state-demo
```

Run an episode with Jev choosing an action every eight emulator frames:

```bash
.venv/bin/typesafe-galaga play --rom "roms/Galaga (U).nes" --frames-per-decision 8
```

Use `--policy heuristic` to run the local heuristic instead of Jev. The display can be
set to `dashboard` (default), `game`, or `none`. Each decision is logged as JSONL under
`artifacts/`.

The parsed JSON groups observations into `player`, `tactical`, `hazards`,
`reaction_timing`, `recent_control`, `episode`, and `audio` (semantic sound event flags).
The debug snapshot additionally includes sprite details and a radar grid (`A` player,
`|` player bullet, `E` enemy, `D` diving enemy, `!` enemy projectile).

## Reinforcement learning

The Gymnasium environment encodes each snapshot as a 164-dimensional `float32` vector
in `[-1.0, 1.0]`:

- **Player & tactical summary (15):** normalized position, velocity, alive status, active
  bullets, lives, danger/evasion cues, enemy/projectile counts, target delta, bullet positions
- **Previous action one-hot (4):** `noop`, `left`, `right`, `fire`
- **Top 6 enemies (24):** presence, `dx`, `dy`, diving flag
- **Top 4 projectiles (12):** presence, `dx`, `dy`
- **9x11 radar occupancy grid (99)**
- **Semantic audio event flags (10, indices `154..163`):** `laser_shot`, `enemy_hit`,
  `boss_hit`, `boss_destroy`, `dive_swoop`, `ship_explode`, `one_up`, `stage_clear`,
  `stage_start`, `ambient`

Instead of rendering audio waveforms during headless training, audio event flags are read
directly from Galaga's live NES RAM sound engine (`$0580-$06E7`) across every frame of a
decision step (capturing both newly triggered and currently active sounds), with automatic
snapshot-transition fallback when RAM is unavailable. See [docs/audio_rl.md](docs/audio_rl.md)
for the full architecture, slot mapping, and guide for custom games.

Audio observations are **enabled by default** (`--audio-obs`). Pass `--no-audio-obs` to
`rl-train` or `rl-bc` to use the legacy 154-dim observation vector; `rl-eval` and
`rl-play` automatically detect the observation dimension from the saved model. The PPO
reward uses score gain, survival, safe firing alignment, lost lives, death, and stage
clear.

All RL commands (`rl-bc`, `rl-train`, `rl-eval`, `rl-play`) default to `--device cpu`,
which is recommended for the `MlpPolicy` network. Pass `--device cuda` (or
`--device auto`) to run on a GPU instead.

### Benchmark: Audio-aware (164-dim) vs Legacy (154-dim)

Evaluated under identical conditions (20 deterministic episodes, `seed=1000`,
`frames_per_decision=4`, `random_noop_max=4`, trained for `1,500,000` timesteps):

| Model | Obs Dim | Mean Score | Min Score | Max Score | Mean Reward (± std) | Mean Steps |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `models/galaga_ppo_best.zip` (legacy) | 154 | `17,624.0` | `10,160` | `21,600` | `156.16 ± 48.03` | `1,729.5` |
| `models/galaga_ppo_audio_best.zip` | 164 | `18,698.0` (+6.1%) | `10,290` | `24,530` | `164.77 ± 67.15` | `1,729.0` |
| `models/galaga_ppo_audio.zip` | 164 | **`18,945.5`** (+7.5%) | **`10,490`** | **`25,000`** | **`169.83 ± 65.19`** | **`1,971.9`** |

### Behavior cloning

Pretrain a PPO-compatible policy from Jev JSONL decision logs (164-dim by default, or add
`--no-audio-obs` for 154-dim):

```bash
.venv/bin/typesafe-galaga rl-bc \
  --inputs artifacts --model-path models/galaga_bc.zip --epochs 30
```

### Training and evaluation

Train an audio-aware PPO agent in parallel emulator workers. Checkpoints go to
`models/checkpoints/`, the final model to `models/galaga_ppo_audio.zip`, and the best
checkpoint to `models/galaga_ppo_audio_best.zip`:

```bash
.venv/bin/typesafe-galaga rl-train \
  --rom "roms/Galaga (U).nes" --timesteps 1500000 --num-envs 8 \
  --model-path models/galaga_ppo_audio.zip --device cuda
```

Resume from a saved audio-aware policy with `--resume-from`:

```bash
.venv/bin/typesafe-galaga rl-train \
  --rom "roms/Galaga (U).nes" --resume-from models/galaga_ppo_audio.zip \
  --model-path models/galaga_ppo_audio.zip
```

To resume a legacy 154-dim model without audio flags, pass `--no-audio-obs`:

```bash
.venv/bin/typesafe-galaga rl-train \
  --rom "roms/Galaga (U).nes" --resume-from models/galaga_ppo_best.zip \
  --no-audio-obs
```

Evaluate any trained model (observation dimension is auto-detected):

```bash
.venv/bin/typesafe-galaga rl-eval \
  --model models/galaga_ppo_audio.zip --rom "roms/Galaga (U).nes" \
  --episodes 20 --output artifacts/eval_galaga_ppo_audio.json
```

### Playing and hybrid mode

Watch a trained PPO policy in the telemetry dashboard (works with both 164-dim audio and
legacy 154-dim models):

```bash
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_audio.zip --rom "roms/Galaga (U).nes"
```

Add `--hybrid` to consult Jev when the RL policy has low confidence or detects a
high-risk situation. Hybrid mode requires `TYPESAFE_API_KEY`:

```bash
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_audio.zip --rom "roms/Galaga (U).nes" \
  --hybrid --confidence-threshold 0.55
```

To run on a GPU instead of the default CPU, pass `--device cuda` (works on `rl-bc`,
`rl-train`, `rl-eval`, and `rl-play`):

```bash
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_audio.zip --rom "roms/Galaga (U).nes" \
  --hybrid --confidence-threshold 0.55 --device cuda
```

## Development

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```
