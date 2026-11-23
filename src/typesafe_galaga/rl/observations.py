from __future__ import annotations

import numpy as np

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

# 15 player/tactical + 4 prev_action + 24 enemies (6*4) + 12 projectiles (4*3) + 99 radar grid = 154
SCALAR_FEATURE_DIM = 15 + NUM_ACTIONS + (MAX_OBSERVED_ENEMIES * 4) + (MAX_OBSERVED_PROJECTILES * 3)
OBSERVATION_DIM = SCALAR_FEATURE_DIM + GRID_CELLS

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


def encode_observation(snapshot: GalagaSnapshot) -> np.ndarray:
    """Convert a GalagaSnapshot into a bounded float32 vector in [-1.0, 1.0]."""
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
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

    return obs
