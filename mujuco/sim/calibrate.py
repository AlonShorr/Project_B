"""m3: run many commands, tabulate discrete outcomes, fit probabilities, write YAML."""

from __future__ import annotations

import collections

import numpy as np
import yaml
from scipy import stats

from .noisy_env import NoisyMazeSim
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from motion_model import DIR_DELTAS, Action, can_step

CARDINAL = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
DIAGONAL = {"NE": (-1, 1), "SE": (1, 1), "SW": (1, -1), "NW": (-1, -1)}

# Wall channel crossed when leaving a cell in a cardinal direction.
DIR_CHANNEL = {(-1, 0): WALL_UP, (1, 0): WALL_DOWN, (0, -1): WALL_LEFT, (0, 1): WALL_RIGHT}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(mid - half), float(mid + half)


def dirichlet_summary(counts: dict[str, int], alpha0: float = 0.5, q: float = 0.95) -> dict:
    """Posterior mean and central credible interval per outcome under a symmetric
    Dirichlet(alpha0) prior (0.5 = Jeffreys). Each marginal is Beta(a_i, sum(a) - a_i).
    'collision' is always in the support so a never-seen collision still gets an upper bound."""
    counts = {"collision": 0, **counts}
    keys = sorted(counts)
    a = np.array([counts[k] + alpha0 for k in keys], float)
    if len(keys) == 1:                     # only 'collision' (a blocked move): p = 1 exactly;
        return {keys[0]: {"p": 1.0, "ci": [1.0, 1.0], "n": int(counts[keys[0]])}}   # Beta(a, 0) is undefined
    out = {}
    for key, ai in zip(keys, a):
        lo, hi = stats.beta.ppf([(1 - q) / 2, (1 + q) / 2], ai, a.sum() - ai)
        out[key] = {"p": float(ai / a.sum()), "ci": [float(lo), float(hi)], "n": int(counts[key])}
    return out


def outcome_label(start: tuple[int, int], end: tuple[int, int], collided: bool) -> str:
    """'collision', or the grid displacement 'moved_<drow>_<dcol>'."""
    if collided:
        return "collision"
    dr, dc = end[0] - start[0], end[1] - start[1]
    return f"moved_{dr:+d}_{dc:+d}"


def context(wall_map: np.ndarray, cell: tuple[int, int], d: tuple[int, int], k: int) -> str:
    """State context RELATIVE TO THE ACTION, for the state-dependence check (cardinal moves):
    'blocked'   - the map itself stops the move within k cells (outcome is deterministic, not noise)
    'side_wall' - path is free but runs along at least one wall (drift ends in a bump)
    'open'      - free path with no side walls.
    Diagonals (k = 1 only) are 'blocked' when motion_model.can_step (D2) forbids them,
    else 'open': with no corner posts a legal diagonal has no wall at its corner."""
    if d not in DIR_CHANNEL:
        if k != 1:
            raise ValueError("diagonal contexts are defined for k = 1 only")
        return "open" if can_step(wall_map, cell[0], cell[1], Action(DIR_DELTAS.index(d))) else "blocked"
    side = [ch for dd, ch in DIR_CHANNEL.items() if dd != d and dd != (-d[0], -d[1])]
    r, c = cell
    has_side = False
    for _ in range(k):
        if wall_map[r, c, DIR_CHANNEL[d]] == 0:      # 0 = wall
            return "blocked"
        r, c = r + d[0], c + d[1]
        has_side |= any(wall_map[r, c, ch] == 0 for ch in side)
    return "side_wall" if has_side else "open"


def run_batch(sim: NoisyMazeSim, wall_map: np.ndarray, actions: dict[str, tuple[int, int]],
              k: int, n: int, rng: np.random.Generator, new_run_every: int = 50) -> dict:
    """n commands per action from random interior cells; returns counts by action and context."""
    rows, cols = wall_map.shape[:2]
    counts: dict = collections.defaultdict(collections.Counter)
    for name, d in actions.items():
        for i in range(n):
            if i % new_run_every == 0:
                sim.new_run()                         # new friction draw
            cell = (int(rng.integers(1, rows - 1)), int(rng.integers(1, cols - 1)))
            sim.teleport(cell)
            res = sim.step(d, k)
            lab = outcome_label(cell, res.cell, res.collided)
            ctx = context(wall_map, cell, d, k)
            counts[(name, ctx)][lab] += 1
            if ctx != "blocked":                      # the map, not noise, decides blocked moves
                counts[(name, "free")][lab] += 1
    return counts


def p_miss(sim: NoisyMazeSim, wall_map: np.ndarray, n: int, rng: np.random.Generator) -> dict:
    """Sensing errors per ray: how often the observed digit differs from the true one.

    A miss is a ray whose reading is wrong (the sensor never returns 'nothing'; see
    noise.py). Returns the overall rate ``p`` with a Wilson interval and the confusion
    counts ``confusion[true][observed]`` (digits 0..R, R + 1 = no wall within R), so a
    sensing model can be fitted to the full error pattern, not only to its rate."""
    rows, cols = wall_map.shape[:2]
    size = sim.R + 2
    confusion = np.zeros((size, size), dtype=int)
    for _ in range(n):
        sim.teleport((int(rng.integers(0, rows)), int(rng.integers(0, cols))))
        for t, o in zip(sim.true_digits(), sim.observe()):
            confusion[t, o] += 1
    tot = int(confusion.sum())
    miss = tot - int(np.trace(confusion))
    lo, hi = wilson(miss, tot)
    return {"p": miss / max(tot, 1), "ci": [lo, hi], "n": tot,
            "confusion": {int(t): {int(o): int(confusion[t, o]) for o in range(size) if confusion[t, o]}
                          for t in range(size) if confusion[t].any()}}


def write_yaml(path: str, motion: dict, miss: dict, meta: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump({"meta": meta, "motion": motion, "p_miss": miss}, f, sort_keys=False)
