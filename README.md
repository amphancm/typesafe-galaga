# TypeSafe Galaga (`typesafe-galaga`)

An experimental controller that lets TypeSafe's Jev model—or a local reinforcement
learning (PPO) agent—choose NES controller inputs for **Galaga**.

The controllers do not receive screenshots. The harness parses emulator RAM and OAM
sprite data into structured state containing the player's position, bullets, enemies,
projectiles, tactical danger cues, score, lives, and stage. Jev receives this state as
JSON; the RL policy receives a bounded 154-dimensional numeric observation. Each
controller action advances the emulator before the next decision.

```text
NES emulator -> RAM/OAM parser -> structured JSON -> Jev Choice -> controller input
                               -> 154-dim vector  -> PPO Policy -> controller input
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

The parsed JSON groups observations into player, tactical, hazards, reaction-timing,
recent-control, and episode data. The debug snapshot additionally includes sprite
details and a radar grid (`A` player, `|` player bullet, `E` enemy, `D` diving enemy,
`!` enemy projectile).

## Reinforcement learning

The Gymnasium environment encodes each snapshot as a 154-dimensional `float32` vector
in `[-1.0, 1.0]`: player and tactical features (15), previous action (4), up to six
enemies (24), up to four projectiles (12), and the 9x11 radar grid (99). The PPO reward
uses score gain, survival, safe firing alignment, lost lives, death, and stage clear.

All RL commands (`rl-bc`, `rl-train`, `rl-eval`, `rl-play`) default to `--device cpu`,
which is recommended for the `MlpPolicy` network. Pass `--device cuda` (or
`--device auto`) to run on a GPU instead.

### Behavior cloning

Pretrain a PPO-compatible policy from Jev JSONL decision logs:

```bash
.venv/bin/typesafe-galaga rl-bc \
  --inputs artifacts --model-path models/galaga_bc.zip --epochs 30
```

### Training and evaluation

Train a PPO agent in parallel emulator workers. Checkpoints go to
`models/checkpoints/`, the final model to `models/galaga_ppo.zip`, and the best model to
`models/galaga_ppo_best.zip`:

```bash
.venv/bin/typesafe-galaga rl-train \
  --rom "roms/Galaga (U).nes" --timesteps 1500000 --num-envs 8 \
  --model-path models/galaga_ppo.zip --device cuda
```

Resume from a saved policy with `--resume-from`:

```bash
.venv/bin/typesafe-galaga rl-train \
  --rom "roms/Galaga (U).nes" --resume-from models/galaga_bc.zip
```

Evaluate a trained model:

```bash
.venv/bin/typesafe-galaga rl-eval \
  --model models/galaga_ppo_best.zip --rom "roms/Galaga (U).nes" \
  --episodes 20 --output artifacts/eval_galaga_ppo_best.json
```

### Playing and hybrid mode

Watch a trained PPO policy in the telemetry dashboard:

```bash
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_best.zip --rom "roms/Galaga (U).nes"
```

Add `--hybrid` to consult Jev when the RL policy has low confidence or detects a
high-risk situation. Hybrid mode requires `TYPESAFE_API_KEY`:

```bash
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_best.zip --rom "roms/Galaga (U).nes" \
  --hybrid --confidence-threshold 0.55
```

To run on a GPU instead of the default CPU, pass `--device cuda` (works on `rl-bc`,
`rl-train`, `rl-eval`, and `rl-play`):

```bash
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_best.zip --rom "roms/Galaga (U).nes" \
  --hybrid --confidence-threshold 0.55 --device cuda
```

## Development

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```
