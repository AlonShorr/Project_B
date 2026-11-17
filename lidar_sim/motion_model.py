"""Project B motion model: 8 absolute directions, noisy step count, drift, collision.

This module is the single source of truth for robot motion. The true robot
(``sample_move`` / ``GridEnv``) samples from it, and the belief filter uses the
matrices from ``build_transition_model``. Both are built on ``resolve_move``,
so the simulator and the filter cannot disagree.

Conventions (shared with the rest of lidar_sim):
- ``wall_map`` has shape ``(rows, cols, 4)``, channels WALL_UP/DOWN/LEFT/RIGHT,
  ``0 = wall`` and ``1 = open``. Both sides of an edge are stored.
- Row 0 is the top (north) row, so N means ``dr = -1``.
- A cell is indexed in the transition matrices as ``r * cols + c``.

Motion semantics (roadmap D2-D4, §3.2):
- The robot commands one of 8 directions. With probability ``1 - p_drift`` it
  moves in that direction, and with ``p_drift / 2`` each it moves 45° to either
  side.
- Independently, it travels ``k`` cells with probability ``p_steps[k]``,
  ``k in {0, 1, 2}``, in a straight line along the realized direction.
- Every edge crossed is checked. A diagonal step goes in a straight line through
  the corner shared by four cells, so it is legal only if all four edges that
  meet at that corner are open. This is the same as requiring both L-paths
  (e.g. N-then-E and E-then-N) to be open.
- ``wall_mode="collide"``: a blocked step ends the run as a COLLISION.
  ``wall_mode="stop"``: the robot stops in front of the wall (Project A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Sequence

import numpy as np

import config
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from discrete_noise import forward_transition_probs

WALL_MODES = ("collide", "stop")


class Action(IntEnum):
    """Actions available to the robot: 8 moves and the terminal declaration."""

    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7
    DECLARE = 8

    @property
    def is_move(self) -> bool:
        return self is not Action.DECLARE

    @property
    def is_diagonal(self) -> bool:
        return self.is_move and int(self) % 2 == 1


MOVE_ACTIONS: tuple[Action, ...] = tuple(a for a in Action if a.is_move)
NUM_MOVES = len(MOVE_ACTIONS)

# (dr, dc) per direction; index = Action value. Clockwise starting at north.
DIR_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)


class Outcome(Enum):
    """State of an episode after a step."""

    RUNNING = "running"
    SUCCESS = "success"
    WRONG_DECLARATION = "wrong_declaration"
    COLLISION = "collision"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self is not Outcome.RUNNING

    @property
    def is_failure(self) -> bool:
        return self in FAILURE_OUTCOMES


FAILURE_OUTCOMES = frozenset(
    {Outcome.WRONG_DECLARATION, Outcome.COLLISION, Outcome.TIMEOUT}
)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MotionParams:
    """Noise parameters of the motion model.

    p_steps:   probabilities of travelling 0, 1 or 2 cells.
    p_drift:   probability of drifting +-45° (split equally between the sides).
    wall_mode: "collide" or "stop" (see module docstring).
    """

    p_steps: tuple[float, float, float] = (0.0, 1.0, 0.0)
    p_drift: float = 0.0
    wall_mode: str = "collide"

    def __post_init__(self):
        p_steps = tuple(float(p) for p in self.p_steps)
        if len(p_steps) != 3:
            raise ValueError("p_steps must have exactly three entries (k = 0, 1, 2)")
        if any(p < 0.0 for p in p_steps) or not np.isclose(sum(p_steps), 1.0):
            raise ValueError(f"p_steps must be non-negative and sum to 1, got {p_steps}")
        if not 0.0 <= self.p_drift <= 1.0:
            raise ValueError(f"p_drift must be in [0, 1], got {self.p_drift}")
        if self.wall_mode not in WALL_MODES:
            raise ValueError(f"wall_mode must be one of {WALL_MODES}, got {self.wall_mode!r}")
        object.__setattr__(self, "p_steps", p_steps)
        object.__setattr__(self, "p_drift", float(self.p_drift))

    @classmethod
    def from_config(cls, noisy: bool | None = None) -> "MotionParams":
        """Build parameters from config.py.

        With noise disabled the model is deterministic (one cell, no drift).
        """
        if noisy is None:
            noisy = config.ENABLE_MOTION_NOISE
        if not noisy:
            return cls((0.0, 1.0, 0.0), 0.0, config.MOTION_WALL_MODE)
        if config.MOTION_P_STEPS is not None:
            p_steps = tuple(config.MOTION_P_STEPS)
        else:
            p_steps = forward_transition_probs(noisy=True)
        return cls(p_steps, config.MOTION_P_DRIFT, config.MOTION_WALL_MODE)

    @classmethod
    def deterministic(cls, wall_mode: str = "collide") -> "MotionParams":
        return cls((0.0, 1.0, 0.0), 0.0, wall_mode)

    @classmethod
    def hard(cls, wall_mode: str = "collide") -> "MotionParams":
        return cls(config.MOTION_P_STEPS_HARD, config.MOTION_P_DRIFT, wall_mode)


def direction_probs(action: Action, p_drift: float) -> list[tuple[Action, float]]:
    """Return [(realized direction, probability)] for a commanded move."""
    action = Action(action)
    if not action.is_move:
        raise ValueError("DECLARE has no motion")
    if p_drift <= 0.0:
        return [(action, 1.0)]
    left = Action((int(action) - 1) % NUM_MOVES)
    right = Action((int(action) + 1) % NUM_MOVES)
    return [(action, 1.0 - p_drift), (left, p_drift / 2.0), (right, p_drift / 2.0)]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _in_bounds(wall_map: np.ndarray, r: int, c: int) -> bool:
    rows, cols = wall_map.shape[:2]
    return 0 <= r < rows and 0 <= c < cols


def _vertical_wall(dr: int) -> int:
    return WALL_UP if dr < 0 else WALL_DOWN


def _horizontal_wall(dc: int) -> int:
    return WALL_LEFT if dc < 0 else WALL_RIGHT


def can_step(wall_map: np.ndarray, r: int, c: int, direction: Action) -> bool:
    """Return True if one step from (r, c) in ``direction`` crosses no wall.

    Cardinal: the shared edge must be open.
    Diagonal: the straight line passes through the corner shared by
    (r, c), (r+dr, c), (r, c+dc) and the target, so all four edges that meet
    at that corner must be open.
    """
    dr, dc = DIR_DELTAS[int(direction)]
    tr, tc = r + dr, c + dc
    if not _in_bounds(wall_map, tr, tc):
        return False

    if dr == 0:
        return wall_map[r, c, _horizontal_wall(dc)] != 0
    if dc == 0:
        return wall_map[r, c, _vertical_wall(dr)] != 0

    return bool(
        # Edge between the source and its vertical neighbour (r+dr, c).
        wall_map[r, c, _vertical_wall(dr)] != 0
        # Edge between the source and its horizontal neighbour (r, c+dc).
        and wall_map[r, c, _horizontal_wall(dc)] != 0
        # Edge between the target and (r, c+dc), seen from the target.
        and wall_map[tr, tc, _vertical_wall(-dr)] != 0
        # Edge between the target and (r+dr, c), seen from the target.
        and wall_map[tr, tc, _horizontal_wall(-dc)] != 0
    )


@dataclass(frozen=True)
class ResolvedMove:
    """Deterministic result of walking k cells in a fixed direction."""

    r: int
    c: int
    collided: bool
    cells_moved: int


def resolve_move(
    wall_map: np.ndarray,
    r: int,
    c: int,
    direction: Action,
    k: int,
    wall_mode: str = "collide",
) -> ResolvedMove:
    """Walk up to ``k`` cells from (r, c) in ``direction``, checking every edge.

    On a blocked edge, "collide" reports a collision and "stop" stops in place.
    In both cases (r, c) is the last cell reached before the wall.
    """
    if wall_mode not in WALL_MODES:
        raise ValueError(f"wall_mode must be one of {WALL_MODES}, got {wall_mode!r}")
    dr, dc = DIR_DELTAS[int(direction)]
    for moved in range(k):
        if not can_step(wall_map, r, c, direction):
            return ResolvedMove(r, c, wall_mode == "collide", moved)
        r, c = r + dr, c + dc
    return ResolvedMove(r, c, False, k)


# --------------------------------------------------------------------------
# Probabilities
# --------------------------------------------------------------------------

def move_distribution(
    wall_map: np.ndarray,
    r: int,
    c: int,
    action: Action,
    params: MotionParams,
) -> tuple[dict[tuple[int, int], float], float]:
    """Return ({(r', c'): P(arrive, no collision)}, P(collision)) for one move."""
    dist: dict[tuple[int, int], float] = {}
    p_collision = 0.0
    for direction, p_dir in direction_probs(action, params.p_drift):
        for k, p_k in enumerate(params.p_steps):
            p = p_dir * p_k
            if p == 0.0:
                continue
            res = resolve_move(wall_map, r, c, direction, k, params.wall_mode)
            if res.collided:
                p_collision += p
            else:
                key = (res.r, res.c)
                dist[key] = dist.get(key, 0.0) + p
    return dist, p_collision


def cell_index(r: int, c: int, cols: int) -> int:
    return r * cols + c


def index_to_cell(i: int, cols: int) -> tuple[int, int]:
    return divmod(int(i), cols)


def build_transition_model(
    wall_map: np.ndarray, params: MotionParams
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, C) for all cells and the 8 move actions.

    T[a, s, s'] = P(s' | s, a, no collision), shape (8, N, N)
    C[a, s]     = P(collision | s, a),         shape (8, N)
    Invariant: T[a, s].sum() + C[a, s] == 1.
    """
    rows, cols = wall_map.shape[:2]
    n = rows * cols
    T = np.zeros((NUM_MOVES, n, n), dtype=np.float64)
    C = np.zeros((NUM_MOVES, n), dtype=np.float64)
    for a in MOVE_ACTIONS:
        for r in range(rows):
            for c in range(cols):
                s = cell_index(r, c, cols)
                dist, p_col = move_distribution(wall_map, r, c, a, params)
                for (nr, nc), p in dist.items():
                    T[a, s, cell_index(nr, nc, cols)] += p
                C[a, s] = p_col
    return T, C


# --------------------------------------------------------------------------
# Sampling (the "true" robot)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveResult:
    """One sampled move of the true robot."""

    action: Action
    direction: Action       # realized direction after drift
    k: int                  # sampled step count (0, 1 or 2)
    start: tuple[int, int]
    r: int                  # cell after the move (last cell before the wall on collision)
    c: int
    cells_moved: int
    collided: bool

    @property
    def drifted(self) -> bool:
        return self.direction != self.action


def _sample_index(probs: Sequence[float], rng: np.random.Generator) -> int:
    u = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    # Floating-point slack: return the last index with non-zero mass.
    return max(i for i, p in enumerate(probs) if p > 0.0)


def sample_move(
    wall_map: np.ndarray,
    r: int,
    c: int,
    action: Action,
    params: MotionParams,
    rng: np.random.Generator,
) -> MoveResult:
    """Sample one noisy move from (r, c). All randomness comes from ``rng``."""
    action = Action(action)
    dirs = direction_probs(action, params.p_drift)
    direction = dirs[_sample_index([p for _, p in dirs], rng)][0]
    k = _sample_index(params.p_steps, rng)
    res = resolve_move(wall_map, r, c, direction, k, params.wall_mode)
    return MoveResult(
        action=action,
        direction=direction,
        k=k,
        start=(r, c),
        r=res.r,
        c=res.c,
        cells_moved=res.cells_moved,
        collided=res.collided,
    )


# --------------------------------------------------------------------------
# Size-agnostic wall_map helpers (map_generator.py relies on config globals)
# --------------------------------------------------------------------------

_OPPOSITE = {WALL_UP: WALL_DOWN, WALL_DOWN: WALL_UP, WALL_LEFT: WALL_RIGHT, WALL_RIGHT: WALL_LEFT}
_WALL_DELTA = {WALL_UP: (-1, 0), WALL_DOWN: (1, 0), WALL_LEFT: (0, -1), WALL_RIGHT: (0, 1)}


def open_wall_map(rows: int, cols: int) -> np.ndarray:
    """Return a wall_map with only the outer border closed."""
    wall_map = np.ones((rows, cols, 4), dtype=int)
    wall_map[0, :, WALL_UP] = 0
    wall_map[-1, :, WALL_DOWN] = 0
    wall_map[:, 0, WALL_LEFT] = 0
    wall_map[:, -1, WALL_RIGHT] = 0
    return wall_map


def set_wall(wall_map: np.ndarray, r: int, c: int, wall: int, closed: bool = True) -> None:
    """Close (or open) one edge, keeping both sides of the edge in sync."""
    value = 0 if closed else 1
    wall_map[r, c, wall] = value
    dr, dc = _WALL_DELTA[wall]
    nr, nc = r + dr, c + dc
    if _in_bounds(wall_map, nr, nc):
        wall_map[nr, nc, _OPPOSITE[wall]] = value
