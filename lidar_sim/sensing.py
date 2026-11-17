"""Project B sensing model: range-R wall sensing on 4 cardinal rays, with misses.

This module is the single source of truth for what the robot senses. The true
robot (``sample_observation`` / ``GridEnv``) samples from the same ray table
that ``build_observation_model`` turns into the matrix ``Z``, so the simulator
and the belief filter cannot disagree.

Conventions (shared with motion_model.py):
- ``wall_map`` has shape ``(rows, cols, 4)``, channels WALL_UP/DOWN/LEFT/RIGHT,
  ``0 = wall`` and ``1 = open``. Row 0 is the top (north) row.
- A cell is indexed as ``r * cols + c``, as in ``build_transition_model``.

Observation semantics (roadmap D5-D7, ROADMAP_TECHNICAL.md §3.3):
- The robot has 4 rays, in the order N, E, S, W. No rotation is needed.
- A wall at distance ``j`` on a ray is the closed edge on the far side of the
  j-th cell along the ray, counting the robot's own cell as j = 0. So ``j = 0``
  is a wall on the robot's own cell edge. Only walls with ``j <= R`` are in
  range. The outer border is a wall like any other, and nothing lies beyond it.
- Each wall in range is missed independently with probability ``p_miss``
  (no false positives). A ray reports the distance of the first wall it does
  not miss, or ``NONE`` if it misses every wall in range (or there is none).
  With walls at ``j_1 < j_2 < ...`` in range:
      P(j_k)  = p_miss**(k-1) * (1 - p_miss)
      P(NONE) = p_miss**(number of walls in range)
- Rays are independent given the cell, so ``Z[s, o]`` is the product of the
  four per-ray probabilities.
- One observation is received at reset and after every move that does not
  collide. DECLARE gives no observation (D7).

Encoding: a ray reading is a digit ``v in {0..R}`` (wall distance) or
``v = R + 1`` (NONE). The observation index is ``o = sum_i v_i * (R+2)**i``
with ray i in the order N, E, S, W, so ``|O| = (R+2)**4`` (256 for R = 2).

8 rays (adding diagonals) are not implemented: D5 notes |O| = 65,536 for R = 2,
which is too large for the solver, and the rule for corner posts on diagonal
rays is still an open question for the supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from motion_model import Action, cell_index

# Ray directions in observation order, with their (dr, dc) and wall channel.
RAY_DIRECTIONS: tuple[Action, ...] = (Action.N, Action.E, Action.S, Action.W)
_RAY_STEPS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, WALL_UP),
    (0, 1, WALL_RIGHT),
    (1, 0, WALL_DOWN),
    (0, -1, WALL_LEFT),
)
NUM_RAYS = len(RAY_DIRECTIONS)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SensingParams:
    """Parameters of the sensing model.

    R:      sensing range in cells (walls at distance 0..R are in range).
    p_miss: probability that a wall in range is missed, independently per wall.
    """

    R: int = 2
    p_miss: float = 0.0

    def __post_init__(self):
        if isinstance(self.R, bool) or int(self.R) != self.R or self.R < 0:
            raise ValueError(f"R must be a non-negative integer, got {self.R!r}")
        if not 0.0 <= self.p_miss < 1.0:
            raise ValueError(f"p_miss must be in [0, 1), got {self.p_miss}")
        object.__setattr__(self, "R", int(self.R))
        object.__setattr__(self, "p_miss", float(self.p_miss))

    @property
    def base(self) -> int:
        """Number of values one ray can report: distances 0..R plus NONE."""
        return self.R + 2

    @property
    def none(self) -> int:
        """Ray digit that means "no wall seen in range"."""
        return self.R + 1

    @property
    def num_observations(self) -> int:
        return self.base ** NUM_RAYS

    @classmethod
    def from_config(cls, noisy: bool | None = None) -> "SensingParams":
        """Build parameters from config.py. With noise disabled, p_miss = 0."""
        if noisy is None:
            noisy = config.ENABLE_SENSOR_NOISE
        return cls(config.SENSING_RANGE_R, config.SENSING_P_MISS if noisy else 0.0)

    @classmethod
    def noiseless(cls, R: int | None = None) -> "SensingParams":
        return cls(config.SENSING_RANGE_R if R is None else R, 0.0)


# --------------------------------------------------------------------------
# Observation encoding
# --------------------------------------------------------------------------

def encode_observation(digits, params: SensingParams) -> int:
    """Turn 4 ray digits (N, E, S, W; ``params.none`` = NONE) into an index."""
    digits = [int(v) for v in digits]
    if len(digits) != NUM_RAYS:
        raise ValueError(f"expected {NUM_RAYS} ray digits, got {len(digits)}")
    o = 0
    for i, v in enumerate(digits):
        if not 0 <= v <= params.none:
            raise ValueError(f"ray digit {v} out of range 0..{params.none}")
        o += v * params.base ** i
    return o


def decode_observation(o: int, params: SensingParams) -> tuple[int, ...]:
    """Inverse of ``encode_observation``: index -> (N, E, S, W) ray digits."""
    o = int(o)
    if not 0 <= o < params.num_observations:
        raise ValueError(f"observation {o} out of range 0..{params.num_observations - 1}")
    digits = []
    for _ in range(NUM_RAYS):
        o, v = divmod(o, params.base)
        digits.append(v)
    return tuple(digits)


def describe_observation(o: int, params: SensingParams) -> str:
    """Human-readable form, e.g. ``'N=0 E=- S=2 W=1'`` (``-`` = NONE)."""
    parts = []
    for d, v in zip(RAY_DIRECTIONS, decode_observation(o, params)):
        parts.append(f"{d.name}={'-' if v == params.none else v}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Geometry: which walls each ray passes within range
# --------------------------------------------------------------------------

def ray_walls(wall_map: np.ndarray, R: int) -> np.ndarray:
    """Return the ray table ``W`` of shape ``(N, 4, R + 1)``, dtype bool.

    ``W[s, i, j]`` is True when ray i from cell s meets a closed edge at
    distance j (the far edge of the j-th cell along the ray). The ray passes
    through closed edges, because a missed wall lets it see the next one, and
    stops at the outer border.
    """
    wall_map = np.asarray(wall_map)
    rows, cols = wall_map.shape[:2]
    table = np.zeros((rows * cols, NUM_RAYS, R + 1), dtype=bool)
    for r0 in range(rows):
        for c0 in range(cols):
            s = cell_index(r0, c0, cols)
            for i, (dr, dc, channel) in enumerate(_RAY_STEPS):
                r, c = r0, c0
                for j in range(R + 1):
                    if wall_map[r, c, channel] == 0:
                        table[s, i, j] = True
                    r, c = r + dr, c + dc
                    if not (0 <= r < rows and 0 <= c < cols):
                        break
    return table


def ray_distributions(table: np.ndarray, params: SensingParams) -> np.ndarray:
    """Per-ray reading distribution, shape ``(N, 4, R + 2)``.

    ``P[s, i, v]`` = probability that ray i from cell s reports digit v.
    """
    n, rays, span = table.shape
    if span != params.R + 1:
        raise ValueError(f"ray table has range {span - 1}, params have R = {params.R}")
    probs = np.zeros((n, rays, params.base), dtype=np.float64)
    survive = np.ones((n, rays), dtype=np.float64)   # P(every wall so far was missed)
    for j in range(params.R + 1):
        wall = table[:, :, j]
        probs[:, :, j] = np.where(wall, survive * (1.0 - params.p_miss), 0.0)
        survive = np.where(wall, survive * params.p_miss, survive)
    probs[:, :, params.none] = survive
    return probs


def build_observation_model(wall_map: np.ndarray, params: SensingParams) -> np.ndarray:
    """Return ``Z`` with ``Z[s, o] = P(o | s)``, shape ``(N, (R+2)**4)``.

    Invariant: every row sums to 1. With ``p_miss = 0`` every row is one-hot.
    """
    probs = ray_distributions(ray_walls(wall_map, params.R), params)
    n = probs.shape[0]
    # Build the product over rays so that ray i is digit i (ray 0 least significant).
    Z = probs[:, 0, :]
    for i in range(1, NUM_RAYS):
        Z = (probs[:, i, :, None] * Z[:, None, :]).reshape(n, -1)
    return Z


def noiseless_signatures(wall_map: np.ndarray, R: int) -> np.ndarray:
    """Observation index each cell gives with no misses, shape ``(N,)``.

    Each ray reports the nearest wall within range, or NONE.
    """
    params = SensingParams(R, 0.0)
    table = ray_walls(wall_map, R)
    any_wall = table.any(axis=2)
    first = np.where(any_wall, table.argmax(axis=2), params.none)
    weights = params.base ** np.arange(NUM_RAYS)
    return first @ weights


def signature_ambiguity(wall_map: np.ndarray, R: int) -> dict[str, float]:
    """R-limited ambiguity of a maze (roadmap §3.3, input to t6.1).

    distinct:        number of distinct noiseless observations
    ambiguity:       1 - distinct / N (0 = every cell looks unique)
    largest_group:   size of the biggest set of cells that look the same
    ambiguous_cells: number of cells that share their observation with another
    """
    sig = noiseless_signatures(wall_map, R)
    _, counts = np.unique(sig, return_counts=True)
    n = sig.size
    return {
        "distinct": int(counts.size),
        "ambiguity": 1.0 - counts.size / n,
        "largest_group": int(counts.max()),
        "ambiguous_cells": int(counts[counts > 1].sum()),
    }


# --------------------------------------------------------------------------
# Sampling (the "true" robot)
# --------------------------------------------------------------------------

def sample_observation(
    table: np.ndarray,
    s: int,
    params: SensingParams,
    rng: np.random.Generator,
) -> int:
    """Sample one observation index at cell index ``s``. Randomness from ``rng``.

    ``table`` is ``ray_walls(wall_map, params.R)``; build it once per map.
    """
    digits = []
    for i in range(NUM_RAYS):
        reading = params.none
        for j in np.flatnonzero(table[s, i]):
            if params.p_miss == 0.0 or rng.random() >= params.p_miss:
                reading = int(j)
                break
        digits.append(reading)
    return encode_observation(digits, params)
