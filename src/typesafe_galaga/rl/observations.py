from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..audio import AUDIO_EVENT_NAMES
from ..state import (
    PLAYER_Y,
    PLAYFIELD_MAX_X,
    PLAYFIELD_MIN_X,
    RADAR_COLS,
    RADAR_ROWS,
    GalagaSnapshot,
)
from .actions import NUM_ACTIONS, index_from_action

GRID_CELLS = RADAR_ROWS * RADAR_COLS
MAX_OBSERVED_ENEMIES = 6
MAX_OBSERVED_PROJECTILES = 4
AUDIO_EVENT_DIM = len(AUDIO_EVENT_NAMES)
AUDIO_EVENT_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(AUDIO_EVENT_NAMES)}
_AUDIO_EVENT_ALIASES: dict[str, str] = {
    "death": "ship_explode",
    "challenging_stage": "stage_clear",
    "overworld": "stage_start",
    "galaga": "stage_start",
}

# 15 player/tactical + 4 prev_action + 24 enemies (6*4) + 12 projectiles (4*3) + 99 radar grid = 154
# + 10 semantic audio event flags = 164
SCALAR_FEATURE_DIM = 15 + NUM_ACTIONS + (MAX_OBSERVED_ENEMIES * 4) + (MAX_OBSERVED_PROJECTILES * 3)
BASE_OBSERVATION_DIM = SCALAR_FEATURE_DIM + GRID_CELLS
OBSERVATION_DIM = BASE_OBSERVATION_DIM + AUDIO_EVENT_DIM

_PLAYFIELD_SPAN = float(max(1, PLAYFIELD_MAX_X - PLAYFIELD_MIN_X))
_RADAR_CELL_MAP: dict[str, float] = {
    "E": 0.5,
    "D": 0.8,
    "!": 1.0,
    "|": -0.5,
    "A": -1.0,
}
_ENEMY_OFFSET = 15 + NUM_ACTIONS
_PROJECTILE_OFFSET = _ENEMY_OFFSET + MAX_OBSERVED_ENEMIES * 4


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return float(value)


def observation_dim(*, include_audio: bool = True) -> int:
    """Return observation vector dimension with or without semantic audio event flags."""
    return OBSERVATION_DIM if include_audio else BASE_OBSERVATION_DIM


def infer_include_audio_from_shape(obs_shape: tuple[int, ...] | None) -> bool:
    """Infer whether a saved policy expects audio flags from its observation_space.shape."""
    if obs_shape is None or len(obs_shape) != 1:
        return True
    dim = int(obs_shape[0])
    if dim == BASE_OBSERVATION_DIM:
        return False
    if dim == OBSERVATION_DIM:
        return True
    raise ValueError(
        f"Unsupported observation dimension {dim}; expected {BASE_OBSERVATION_DIM} "
        f"(without audio) or {OBSERVATION_DIM} (with audio)"
    )


def encode_audio_events(audio_events: Sequence[str]) -> np.ndarray:
    """Encode semantic audio event names into a binary float32 vector of length AUDIO_EVENT_DIM."""
    vec = np.zeros(AUDIO_EVENT_DIM, dtype=np.float32)
    for event in audio_events:
        canonical = _AUDIO_EVENT_ALIASES.get(event, event)
        idx = AUDIO_EVENT_INDEX.get(canonical)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def encode_observation(
    snapshot: GalagaSnapshot,
    *,
    include_audio: bool = True,
) -> np.ndarray:
    """Convert a GalagaSnapshot into a bounded float32 vector in [-1.0, 1.0]."""
    obs = np.zeros(observation_dim(include_audio=include_audio), dtype=np.float32)
    sx = snapshot.x
    norm_x = ((sx - PLAYFIELD_MIN_X) / _PLAYFIELD_SPAN) * 2.0 - 1.0

    # --- Player & tactical summary (15) ---
    obs[0] = _clip(norm_x, -1.0, 1.0)
    obs[1] = _clip(snapshot.dx / 16.0, -1.0, 1.0)
    if snapshot.alive:
        obs[2] = 1.0
    obs[3] = _clip(snapshot.active_player_bullets / 2.0, 0.0, 1.0)
    obs[4] = _clip(snapshot.lives / 3.0, 0.0, 1.0)
    if snapshot.danger_imminent:
        obs[5] = 1.0
    evade = snapshot.recommended_evade_dir
    if evade == "left":
        obs[6] = 1.0
    elif evade == "right":
        obs[6] = -1.0

    enemies = snapshot.enemies
    projectiles = snapshot.projectiles
    n_enemies = len(enemies)
    n_projectiles = len(projectiles)

    obs[7] = _clip(n_enemies / 40.0, 0.0, 1.0)
    obs[8] = _clip(snapshot.aligned_enemy_count / 8.0, 0.0, 1.0)
    obs[9] = _clip(n_projectiles / 8.0, 0.0, 1.0)

    if snapshot.best_target_x is not None:
        obs[10] = _clip((snapshot.best_target_x - sx) / 168.0, -1.0, 1.0)

    bullets = snapshot.player_bullets
    n_bullets = len(bullets)
    if n_bullets > 0:
        bx0, by0 = bullets[0]
        obs[11] = _clip((bx0 - sx) / 168.0)
        obs[12] = _clip((PLAYER_Y - by0) / 200.0, 0.0, 1.0)
        if n_bullets > 1:
            bx1, by1 = bullets[1]
            obs[13] = _clip((bx1 - sx) / 168.0)
            obs[14] = _clip((PLAYER_Y - by1) / 200.0, 0.0, 1.0)

    # --- Previous action one-hot (4) ---
    if snapshot.previous_action is not None:
        try:
            prev_idx = index_from_action(snapshot.previous_action)
        except ValueError:
            prev_idx = -1
        if 0 <= prev_idx < NUM_ACTIONS:
            obs[15 + prev_idx] = 1.0

    # --- Top 6 enemies (24) ---
    n_obs_enemies = min(MAX_OBSERVED_ENEMIES, n_enemies)
    base = _ENEMY_OFFSET
    for slot in range(n_obs_enemies):
        e = enemies[slot]
        obs[base] = 1.0
        obs[base + 1] = _clip(e.dx_from_player / 168.0, -1.0, 1.0)
        obs[base + 2] = _clip(e.dy_from_player / 200.0, -1.0, 1.0)
        if e.diving:
            obs[base + 3] = 1.0
        base += 4
    for _ in range(n_obs_enemies, MAX_OBSERVED_ENEMIES):
        obs[base + 2] = 1.0
        base += 4

    # --- Top 4 projectiles (12) ---
    n_obs_proj = min(MAX_OBSERVED_PROJECTILES, n_projectiles)
    base = _PROJECTILE_OFFSET
    for slot in range(n_obs_proj):
        p = projectiles[slot]
        obs[base] = 1.0
        obs[base + 1] = _clip(p.dx_from_player / 168.0, -1.0, 1.0)
        obs[base + 2] = _clip(p.dy_from_player / 200.0, -1.0, 1.0)
        base += 3
    for _ in range(n_obs_proj, MAX_OBSERVED_PROJECTILES):
        obs[base + 2] = 1.0
        base += 3

    # --- 9x11 radar occupancy grid (99) ---
    grid = snapshot.radar_grid
    cell_map = _RADAR_CELL_MAP
    row_base = SCALAR_FEATURE_DIM
    n_rows = min(RADAR_ROWS, len(grid))
    for row_idx in range(n_rows):
        row = grid[row_idx]
        n_cols = min(RADAR_COLS, len(row))
        for col_idx in range(n_cols):
            ch = row[col_idx]
            if ch != ".":
                val = cell_map.get(ch)
                if val is not None:
                    obs[row_base + col_idx] = val
        row_base += RADAR_COLS

    if include_audio and snapshot.audio_events:
        obs[BASE_OBSERVATION_DIM:] = encode_audio_events(snapshot.audio_events)

    return obs
