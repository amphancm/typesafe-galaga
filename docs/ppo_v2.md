# PPO v2 (`galaga_ppo_v2`) Architecture & Training Guide

`galaga_ppo_v2` is the second-generation reinforcement learning pipeline for
`typesafe-galaga`. It raises mean deterministic evaluation score from `18,945.5`
(`galaga_ppo_audio.zip`) to **`31,269.5`** (+65.0% over v1 audio, +77.4% over the v1
legacy baseline), with a peak single-episode score of **`51,370`** and a minimum score of
**`20,930`** across 20 deterministic episodes.

## Summary of Changes from v1 to v2

| Component | v1 (`galaga_ppo_audio`) | v2 (`galaga_ppo_v2`) |
| --- | --- | --- |
| Action space | 4 (`noop`, `left`, `right`, `fire`) | **6** (+ `left_fire`, `right_fire`) |
| Single-step obs dim | 164 (154 base + 10 audio) | **166** (156 base + 10 audio) |
| Frame stacking | None (`1`) | **4 steps** (`664`-dim total) |
| Radar cell priority | Player bullets could overwrite enemies/projectiles | **Threat-prioritized** (`\|` < `E` < `D` < `!` < `A`) |
| Lives tracking | Sprite-absence counter starting at 3 | **Live NES RAM `$0487`** (supports 1-UPs & custom ROMs) |
| Score reward cap | `10.0` (clipped >1,000 pt bonuses) | **`100.0`** (preserves Challenging Stage bonuses) |
| Stage clear detection | `clear` flag only | **Stage increment + `stage_clear` audio rising edge** (50-step cooldown) |
| Policy / value MLP | Shared `[256, 256]` | **Asymmetric** `pi=[256, 256]`, `vf=[512, 512]` |
| Reward normalization | Off | **`VecNormalize`** (`norm_obs=False`, `norm_reward=True`, `clip_reward=10.0`) |
| Rollout & discount | `n_steps=512`, `gamma=0.99`, `ent_coef=0.015` | **`n_steps=1024`**, **`gamma=0.995`**, **`ent_coef=0.01`** |
| Training budget | `1,500,000` timesteps | **`5,000,000`** timesteps |
| Best checkpointing | Rolling training episode score | **Periodic deterministic eval** (`GalagaEvalCallback`, every 200k steps) |

## 1. Six-Action Move-and-Shoot Space

In v1, firing forced the ship to stop moving (`["A"]`), making it difficult to dodge diving
enemies and projectiles while maintaining fire rate. v2 adds simultaneous strafe-and-fire
actions in `src/typesafe_galaga/actions.py`:

| Index | Action | NES Buttons | Fire Release Action |
| :---: | --- | --- | --- |
| `0` | `noop` | `["NOOP"]` | — |
| `1` | `left` | `["left"]` | — |
| `2` | `right` | `["right"]` | — |
| `3` | `fire` | `["A"]` | `noop` |
| `4` | `left_fire` | `["left", "A"]` | `left` |
| `5` | `right_fire` | `["right", "A"]` | `right` |

Galaga requires releasing the `A` button between shots. `FIRE_RELEASE_ACTION` releases `A`
while preserving horizontal movement (`left_fire` → `left`, `right_fire` → `right`).

## 2. Observation Space & 4-Step Frame Stacking

Expanding to 6 actions grows the previous-action one-hot from 4 to 6 features, bringing the
single-step observation vector to **166 dimensions** (`156` base + `10` semantic audio
flags; see [docs/audio_rl.md](audio_rl.md) for the audio flag layout):

- `15` player & tactical summary features
- `6` previous-action one-hot features
- `24` top-6 enemies (`presence`, `dx`, `dy`, `diving`)
- `12` top-4 enemy projectiles (`presence`, `dx`, `dy`)
- `99` `9×11` radar occupancy grid
- `10` semantic audio event flags (indices `156..165`)

During training, `VecFrameStack(vec_env, n_stack=4)` concatenates the last **4 decision
steps** (`16` emulator frames at `frames_per_decision=4`) into a **664-dimensional**
vector (`624`-dim if `--no-audio-obs` is passed). At evaluation, playback, and behavior
cloning time, `FrameStackBuffer` in `src/typesafe_galaga/rl/observations.py` reproduces the
exact same SB3 stacking layout.

### Auto-Detection at Eval & Play Time

`infer_obs_config_from_shape()` inspects `model.observation_space.shape` when loading a
checkpoint in `rl-eval` or `rl-play`:

- Multiples of `166` (e.g. `(166,)`, `(664,)`) enable audio observations and set
  `frame_stack = dim // 166`.
- Multiples of `156` (e.g. `(156,)`, `(624,)`) disable audio observations and set
  `frame_stack = dim // 156`.

`RLPolicy.reset_episode()` and `HybridPolicy.reset_episode()` clear the `FrameStackBuffer`
at the start of every episode and on dashboard restarts.

## 3. State & Environment Improvements

- **Threat-prioritized radar grid (`src/typesafe_galaga/state.py`):** `_build_radar_grid()`
  draws cells in ascending threat priority: player bullets (`|`) → formation enemies (`E`)
  → diving enemies (`D`) → enemy projectiles (`!`) → player ship (`A`). Player bullets can
  no longer mask incoming projectiles or diving enemies occupying the same grid cell.
- **RAM-backed lives tracking (`src/typesafe_galaga/runner.py`):** `GalagaEnvWrapper`
  reads reserve lives directly from NES RAM `$0487` (`LIVES_RESERVE_ADDR`). This keeps
  lives accurate across 1-UP extra lives, final respawns, and custom ROMs (such as
  30-lives variants).

## 4. Reward Shaping & Stage-Clear Progression

Updated defaults in `RewardConfig` (`src/typesafe_galaga/rl/rewards.py`):

| Parameter | v1 | v2 | Purpose |
| --- | :---: | :---: | --- |
| `score_scale` | `0.01` | `0.01` | `+1.0` reward per 100 game points |
| `score_bonus_cap` | `10.0` | **`100.0`** | Avoid clipping Challenging Stage bonuses (e.g. 5,000–10,000 pts) |
| `survival_bonus` | `0.01` | **`0.005`** | Reduce passive survival farming relative to scoring |
| `alignment_bonus` | `0.02` | **`0.01`** | Lighter shaping for lining up with targets |
| `fire_when_aligned_bonus` | `0.05` | **`0.04`** | Encourage firing when safely aligned |
| `life_lost_penalty` | `-10.0` | `-10.0` | Penalty on losing a ship |
| `death_penalty` | `-20.0` | `-20.0` | Terminal game-over penalty |
| `stage_clear_bonus` | `25.0` | `25.0` | Stage-clear progression bonus |
| `stage_clear_cooldown_steps` | — | **`50`** | Prevent double-triggering between stage increment and audio cue |

`GalagaRewardCalculator` now awards `stage_clear_bonus` on either a stage number increase
(`current.stage > _highest_stage`) or a rising edge on the `stage_clear` semantic audio
event, suppressed by `stage_clear_cooldown_steps` to prevent double counting.

## 5. PPO Hyperparameters & Training Pipeline

Configured in `TrainConfig` (`src/typesafe_galaga/rl/train.py`):

- **Timesteps:** `5,000,000` across `8` parallel `SubprocVecEnv` workers (`frames_per_decision=4`,
  `max_steps=4500`, `random_noop_max=4`).
- **Network architecture:** Asymmetric MLP with `pi=[256, 256]` and `vf=[512, 512]`,
  giving the critic extra capacity to model long-horizon multi-stage returns.
- **PPO optimization:** Linear learning-rate decay `3e-4 → 3e-5`, `n_steps=1024`,
  `batch_size=256`, `n_epochs=10`, `gamma=0.995`, `gae_lambda=0.95`, `clip_range=0.2`,
  `ent_coef=0.01`.
- **Reward normalization:** `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.995)`
  stabilizes value targets across large Challenging Stage score spikes while leaving the
  bounded `[-1, 1]` observation vector untouched (so no running observation stats are
  needed at inference time).
- **Deterministic evaluation callback (`GalagaEvalCallback`):** Every `--eval-freq`
  timesteps (default `200,000`), evaluates the policy deterministically for
  `--eval-episodes` episodes (default `10`, `seed=1000..1009`) on a dedicated
  `DummyVecEnv` + `VecFrameStack` and saves `<model_stem>_best.zip` whenever mean
  deterministic score improves.

## 6. Benchmark Results

All models evaluated on `roms/Galaga (U).nes` across 20 deterministic episodes
(`seed=1000`, `frames_per_decision=4`, `random_noop_max=4`):

| Model | Actions | Obs Dim | Steps | Mean Score | Min Score | Max Score | Mean Reward (± std) | Mean Steps |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `models/galaga_ppo_best.zip` (v1 legacy) | 4 | 154 | 1.5M | `17,624.0` | `10,160` | `21,600` | `156.16 ± 48.03` | `1,729.5` |
| `models/galaga_ppo_audio_best.zip` (v1) | 4 | 164 | 1.5M | `18,698.0` (+6.1%) | `10,290` | `24,530` | `164.77 ± 67.15` | `1,729.0` |
| `models/galaga_ppo_audio.zip` (v1) | 4 | 164 | 1.5M | `18,945.5` (+7.5%) | `10,490` | `25,000` | `169.83 ± 65.19` | `1,971.9` |
| `models/galaga_ppo_v2_best.zip` (v2) | 6 | 664 (`166×4`) | 4.6M | `25,053.5` (+42.2%) | `10,250` | `37,770` | `223.28 ± 89.96` | `2,378.0` |
| `models/galaga_ppo_v2.zip` (v2) | 6 | 664 (`166×4`) | 5.0M | **`31,269.5`** (+77.4%) | **`20,930`** | **`51,370`** | **`287.57 ± 141.78`** | **`2,831.0`** |

> [!NOTE]
> `galaga_ppo_v2_best.zip` was saved at `4,600,000` steps based on the 10-episode
> intermediate evaluation, while the final `5,000,000`-step model (`galaga_ppo_v2.zip`)
> achieved the highest 20-episode evaluation score (`31,269.5` mean, `51,370` max).

## 7. Reproducing Training, Evaluation, and Playback

```bash
# Train PPO v2 (5M timesteps, 8 envs, 4-step frame stack, audio obs, reward norm)
.venv/bin/python -m typesafe_galaga.rl.train \
  --rom "roms/Galaga (U).nes" \
  --timesteps 5000000 \
  --num-envs 8 \
  --model-path models/galaga_ppo_v2.zip \
  --checkpoint-dir models/checkpoints_v2 \
  --tensorboard-log runs/galaga_ppo_v2 \
  --device cuda

# Evaluate over 20 deterministic episodes
.venv/bin/typesafe-galaga rl-eval \
  --model models/galaga_ppo_v2.zip \
  --rom "roms/Galaga (U).nes" \
  --episodes 20 \
  --seed 1000 \
  --output artifacts/eval_galaga_ppo_v2.json

# Watch in the live dashboard
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_v2.zip \
  --rom "roms/Galaga (U).nes"

# Watch in hybrid mode (PPO v2 + Jev escalation, requires TYPESAFE_API_KEY)
.venv/bin/typesafe-galaga rl-play \
  --model models/galaga_ppo_v2.zip \
  --rom "roms/Galaga (U).nes" \
  --hybrid --confidence-threshold 0.55
```
