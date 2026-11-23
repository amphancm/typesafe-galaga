from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

PLAYFIELD_MIN_X = 16
PLAYFIELD_MAX_X = 184
PLAYER_Y = 199
OAM_START = 0x0700
OAM_END = 0x0800
PLAYER_X_ADDR = 0x0203
SCORE_DIGITS_ADDR = 0x00E0
SCORE_DIGITS_LEN = 7

TILE_PLAYER_SHIP = 0x00
TILE_PLAYER_BULLET = 0xF6
TILE_STAR = 0xFE

RADAR_ROWS = 9
RADAR_COLS = 11
_PLAYFIELD_SPAN = max(1, PLAYFIELD_MAX_X - PLAYFIELD_MIN_X)
_RADAR_MAX_ROW = RADAR_ROWS - 1
_RADAR_MAX_COL = RADAR_COLS - 1


@dataclass(frozen=True, slots=True)
class GalagaEnemy:
    x: int
    y: int
    dx_from_player: int
    dy_from_player: int
    diving: bool
    aligned: bool
    tile: int = 0

    def to_state(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "dx_from_player": self.dx_from_player,
            "dy_from_player": self.dy_from_player,
            "diving": self.diving,
            "aligned": self.aligned,
        }


@dataclass(frozen=True, slots=True)
class GalagaProjectile:
    x: int
    y: int
    dx_from_player: int
    dy_from_player: int

    def to_state(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "dx_from_player": self.dx_from_player,
            "dy_from_player": self.dy_from_player,
        }


@dataclass(frozen=True, slots=True)
class GalagaSnapshot:
    goal: str
    stage: int
    score: int
    lives: int
    x: int
    y: int
    dx: int
    alive: bool
    dead: bool
    clear: bool
    active_player_bullets: int
    player_bullets: tuple[tuple[int, int], ...] = ()
    enemies: tuple[GalagaEnemy, ...] = ()
    projectiles: tuple[GalagaProjectile, ...] = ()
    nearest_diving_enemy: GalagaEnemy | None = None
    nearest_projectile: GalagaProjectile | None = None
    aligned_enemy_count: int = 0
    danger_imminent: bool = False
    recommended_evade_dir: str = "none"
    best_target_x: int | None = None
    radar_grid: tuple[str, ...] = ()
    previous_action: str | None = None
    previous_reward: float = 0.0
    decision_horizon_frames: int = 8
    response_delay_frames: int = 0
    response_latency_ms: float = 0.0
    raw_info: Mapping[str, Any] = field(default_factory=dict)

    def threat_features(self) -> dict[str, Any]:
        diving_enemies = [e for e in self.enemies if e.diving]
        nearest_diver = self.nearest_diving_enemy
        nearest_proj = self.nearest_projectile
        best_target_dx = (self.best_target_x - self.x) if self.best_target_x is not None else 0
        return {
            "enemy_count": len(self.enemies),
            "diving_count": len(diving_enemies),
            "projectile_count": len(self.projectiles),
            "aligned_enemy_count": self.aligned_enemy_count,
            "danger_imminent": self.danger_imminent,
            "recommended_evade_dir": self.recommended_evade_dir,
            "best_target_x": self.best_target_x,
            "best_target_dx": best_target_dx,
            "nearest_diver_dx": nearest_diver.dx_from_player if nearest_diver else None,
            "nearest_diver_dy": nearest_diver.dy_from_player if nearest_diver else None,
            "nearest_proj_dx": nearest_proj.dx_from_player if nearest_proj else None,
            "nearest_proj_dy": nearest_proj.dy_from_player if nearest_proj else None,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "player": {
                "x": self.x,
                "y": self.y,
                "dx": self.dx,
                "alive": self.alive,
                "active_bullets": self.active_player_bullets,
                "can_fire_more": self.active_player_bullets < 2,
                "playfield_bounds": [PLAYFIELD_MIN_X, PLAYFIELD_MAX_X],
            },
            "tactical": {
                "enemy_count": len(self.enemies),
                "aligned_enemy_count": self.aligned_enemy_count,
                "best_target_x": self.best_target_x,
                "best_target_dx": (
                    (self.best_target_x - self.x) if self.best_target_x is not None else None
                ),
                "danger_imminent": self.danger_imminent,
                "recommended_evade_dir": self.recommended_evade_dir,
            },
            "hazards": {
                "diving_enemies": [e.to_state() for e in self.enemies if e.diving][:4],
                "projectiles": [p.to_state() for p in self.projectiles][:4],
            },
            "reaction_timing": {
                "action_hold_frames": self.decision_horizon_frames,
                "last_response_delay_frames": self.response_delay_frames,
                "last_response_latency_ms": round(self.response_latency_ms, 1),
            },
            "recent_control": {
                "previous_action": self.previous_action,
                "previous_reward": round(self.previous_reward, 3),
            },
            "episode": {
                "stage": self.stage,
                "score": self.score,
                "lives": self.lives,
                "dead": self.dead,
                "clear": self.clear,
            },
        }

    def to_debug_state(self) -> dict[str, Any]:
        state = self.to_state()
        state["enemies"] = [e.to_state() for e in self.enemies[:12]]
        state["player_bullets"] = [{"x": bx, "y": by} for bx, by in self.player_bullets]
        state["radar_grid"] = list(self.radar_grid)
        return state

    def to_text(self) -> str:
        lines = [
            f"Goal: {self.goal}",
            (
                f"Ship: x={self.x} y={self.y} dx={self.dx:+d}, alive={self.alive}, "
                f"bullets={self.active_player_bullets}/2"
            ),
            f"Stage={self.stage}, Score={self.score}, Lives={self.lives}",
            (
                f"Enemies={len(self.enemies)} (aligned={self.aligned_enemy_count}), "
                f"Projectiles={len(self.projectiles)}, Danger={self.danger_imminent} "
                f"(evade={self.recommended_evade_dir})"
            ),
        ]
        if self.radar_grid:
            lines.append("Radar (A=ship, |=bullet, E=enemy, D=diver, !=shot):")
            lines.extend(self.radar_grid)
        return "\n".join(lines)


def _read_score(ram: Sequence[int]) -> int:
    if len(ram) < SCORE_DIGITS_ADDR + SCORE_DIGITS_LEN:
        return 0
    total = 0
    try:
        digits = memoryview(ram)[SCORE_DIGITS_ADDR : SCORE_DIGITS_ADDR + SCORE_DIGITS_LEN]
    except TypeError:
        for idx in range(SCORE_DIGITS_ADDR, SCORE_DIGITS_ADDR + SCORE_DIGITS_LEN):
            digit = int(ram[idx])
            if 0 <= digit <= 9:
                total = total * 10 + digit
        return total
    for digit in digits:
        if 0 <= digit <= 9:
            total = total * 10 + digit
    return total


def _parse_oam_entities(
    ram: Sequence[int],
    player_x: int,
) -> tuple[
    bool,
    int,
    tuple[tuple[int, int], ...],
    tuple[GalagaEnemy, ...],
    tuple[GalagaProjectile, ...],
]:
    if len(ram) < OAM_END:
        return True, player_x, (), (), ()

    ship_xs: list[int] = []
    bullets: list[tuple[int, int]] = []
    other_sprites: list[tuple[int, int, int]] = []  # (x, y, tile)

    try:
        oam: Any = memoryview(ram)[OAM_START:OAM_END]
        for offset in range(0, len(oam), 4):
            y = oam[offset]
            tile = oam[offset + 1]
            if y <= 0 or y >= 232 or tile == TILE_STAR:
                continue
            x = oam[offset + 3]
            if tile == TILE_PLAYER_SHIP and abs(y - PLAYER_Y) <= 4:
                ship_xs.append(x)
            elif tile == TILE_PLAYER_BULLET:
                bullets.append((x, y))
            elif x < 192:
                other_sprites.append((x, y, tile))
    except TypeError:
        for offset in range(OAM_START, OAM_END, 4):
            y = int(ram[offset])
            tile = int(ram[offset + 1])
            if y <= 0 or y >= 232 or tile == TILE_STAR:
                continue
            x = int(ram[offset + 3])
            if tile == TILE_PLAYER_SHIP and abs(y - PLAYER_Y) <= 4:
                ship_xs.append(x)
            elif tile == TILE_PLAYER_BULLET:
                bullets.append((x, y))
            elif x < 192:
                other_sprites.append((x, y, tile))

    alive = bool(ship_xs)
    resolved_x = (sum(ship_xs) // len(ship_xs)) if ship_xs else player_x

    # Pair adjacent 8x16 sprites belonging to the same enemy.
    other_sprites.sort(key=lambda item: (item[1], item[0]))
    n_other = len(other_sprites)
    used = [False] * n_other
    enemies: list[GalagaEnemy] = []
    projectiles: list[GalagaProjectile] = []

    for i in range(n_other):
        if used[i]:
            continue
        x1, y1, t1 = other_sprites[i]
        partner_idx = -1
        for j in range(i + 1, n_other):
            if used[j]:
                continue
            x2, y2, _ = other_sprites[j]
            if y2 - y1 > 2:
                break
            dx_pair = abs(x2 - x1)
            if 4 <= dx_pair <= 10:
                partner_idx = j
                break
        used[i] = True
        if partner_idx != -1:
            used[partner_idx] = True
            x2, y2, _ = other_sprites[partner_idx]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dx = cx - resolved_x
            dy = PLAYER_Y - cy
            enemies.append(
                GalagaEnemy(
                    x=cx,
                    y=cy,
                    dx_from_player=dx,
                    dy_from_player=dy,
                    diving=cy >= 96,
                    aligned=-10 <= dx <= 10,
                    tile=t1,
                )
            )
        else:
            dx = x1 - resolved_x
            dy = PLAYER_Y - y1
            projectiles.append(
                GalagaProjectile(
                    x=x1,
                    y=y1,
                    dx_from_player=dx,
                    dy_from_player=dy,
                )
            )

    enemies.sort(key=lambda e: (-int(e.diving), abs(e.dx_from_player), -e.y))
    projectiles.sort(key=lambda p: (abs(p.dy_from_player), abs(p.dx_from_player)))
    if len(bullets) > 1:
        bullets.sort(key=lambda b: b[1])
    return alive, resolved_x, tuple(bullets[:2]), tuple(enemies), tuple(projectiles)


def _to_radar_cell(x: int, y: int) -> tuple[int, int]:
    col = int((x - PLAYFIELD_MIN_X) / _PLAYFIELD_SPAN * _RADAR_MAX_COL)
    row = int(y / 210.0 * _RADAR_MAX_ROW)
    if row < 0:
        row = 0
    elif row > _RADAR_MAX_ROW:
        row = _RADAR_MAX_ROW
    if col < 0:
        col = 0
    elif col > _RADAR_MAX_COL:
        col = _RADAR_MAX_COL
    return row, col


def _build_radar_grid(
    player_x: int,
    bullets: Sequence[tuple[int, int]],
    enemies: Sequence[GalagaEnemy],
    projectiles: Sequence[GalagaProjectile],
) -> tuple[str, ...]:
    grid = [["."] * RADAR_COLS for _ in range(RADAR_ROWS)]

    for e in enemies:
        r, c = _to_radar_cell(e.x, e.y)
        grid[r][c] = "D" if e.diving else "E"
    for p in projectiles:
        r, c = _to_radar_cell(p.x, p.y)
        grid[r][c] = "!"
    for bx, by in bullets:
        r, c = _to_radar_cell(bx, by)
        grid[r][c] = "|"
    pr, pc = _to_radar_cell(player_x, PLAYER_Y)
    grid[pr][pc] = "A"
    return tuple("".join(row) for row in grid)


class GalagaStateParser:
    """Parse NES Galaga RAM + OAM into a structured GalagaSnapshot."""

    def __init__(
        self,
        *,
        goal: str = "Survive and destroy all Galaga enemies for the highest score.",
        decision_horizon_frames: int = 8,
    ) -> None:
        self.goal = goal
        self.decision_horizon_frames = decision_horizon_frames
        self._last_x: int | None = None
        self._last_score: int = 0

    def reset(self) -> None:
        self._last_x = None
        self._last_score = 0

    def parse(
        self,
        info: Mapping[str, Any] | None,
        ram: Sequence[int] | None,
        *,
        previous_action: str | None = None,
        previous_reward: float = 0.0,
        response_delay_frames: int = 0,
        response_latency_ms: float = 0.0,
    ) -> GalagaSnapshot:
        info = info or {}
        ram_seq: Sequence[int] = ram if ram is not None else ()

        raw_x = int(ram_seq[PLAYER_X_ADDR]) if len(ram_seq) > PLAYER_X_ADDR else 100
        if raw_x == 0:
            raw_x = 100
        alive, x, bullets, enemies, projectiles = _parse_oam_entities(ram_seq, raw_x)

        score = int(info.get("score", _read_score(ram_seq)))
        lives = int(info.get("lives", info.get("life", 3)))
        stage = int(info.get("stage", 1))
        dead = bool(info.get("death", info.get("dead", False)))
        clear = bool(info.get("clear", False))

        dx = 0 if self._last_x is None else x - self._last_x
        self._last_x = x
        self._last_score = max(self._last_score, score)

        nearest_diver: GalagaEnemy | None = None
        nearest_diver_dist = 1_000_000
        aligned_count = 0
        best_target_x: int | None = None
        best_target_key: tuple[bool, int, int] | None = None

        for e in enemies:
            if e.aligned:
                aligned_count += 1
            if e.diving:
                dist = abs(e.dx_from_player) + (max(0, e.dy_from_player))
                if dist < nearest_diver_dist:
                    nearest_diver_dist = dist
                    nearest_diver = e
            tkey = (not e.diving, -e.y, abs(e.dx_from_player))
            if best_target_key is None or tkey < best_target_key:
                best_target_key = tkey
                best_target_x = e.x

        nearest_proj: GalagaProjectile | None = None
        if projectiles:
            nearest_proj_dist = 1_000_000
            for p in projectiles:
                dist = abs(p.dx_from_player) + (max(0, p.dy_from_player))
                if dist < nearest_proj_dist:
                    nearest_proj_dist = dist
                    nearest_proj = p

        # Determine imminent danger and evasion direction.
        threat_dx: int | None = None
        if (
            nearest_proj is not None
            and 0 <= nearest_proj.dy_from_player <= 55
            and abs(nearest_proj.dx_from_player) <= 18
        ):
            threat_dx = nearest_proj.dx_from_player
        elif (
            nearest_diver is not None
            and 0 <= nearest_diver.dy_from_player <= 60
            and abs(nearest_diver.dx_from_player) <= 22
        ):
            threat_dx = nearest_diver.dx_from_player

        danger_imminent = threat_dx is not None
        if threat_dx is None:
            evade_dir = "none"
        elif threat_dx >= 0:
            evade_dir = "left" if x > PLAYFIELD_MIN_X + 16 else "right"
        else:
            evade_dir = "right" if x < PLAYFIELD_MAX_X - 16 else "left"

        radar = _build_radar_grid(x, bullets, enemies, projectiles)

        return GalagaSnapshot(
            goal=self.goal,
            stage=stage,
            score=score,
            lives=lives,
            x=x,
            y=PLAYER_Y,
            dx=dx,
            alive=alive,
            dead=dead,
            clear=clear,
            active_player_bullets=len(bullets),
            player_bullets=bullets,
            enemies=enemies,
            projectiles=projectiles,
            nearest_diving_enemy=nearest_diver,
            nearest_projectile=nearest_proj,
            aligned_enemy_count=aligned_count,
            danger_imminent=danger_imminent,
            recommended_evade_dir=evade_dir,
            best_target_x=best_target_x,
            radar_grid=radar,
            previous_action=previous_action,
            previous_reward=previous_reward,
            decision_horizon_frames=self.decision_horizon_frames,
            response_delay_frames=response_delay_frames,
            response_latency_ms=response_latency_ms,
            raw_info=dict(info),
        )


def with_timing(
    snapshot: GalagaSnapshot,
    *,
    previous_action: str | None = None,
    previous_reward: float | None = None,
    response_delay_frames: int | None = None,
    response_latency_ms: float | None = None,
) -> GalagaSnapshot:
    return replace(
        snapshot,
        previous_action=snapshot.previous_action if previous_action is None else previous_action,
        previous_reward=snapshot.previous_reward if previous_reward is None else previous_reward,
        response_delay_frames=(
            snapshot.response_delay_frames
            if response_delay_frames is None
            else response_delay_frames
        ),
        response_latency_ms=(
            snapshot.response_latency_ms if response_latency_ms is None else response_latency_ms
        ),
    )
