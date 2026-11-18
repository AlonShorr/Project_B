"""Numbers behind the default parameters (t1.2 / t1.3).

Run from lidar_sim/:  ../.venv/Scripts/python.exe defaults_analysis.py
Writes docs/decisions/defaults_analysis.json and prints the tables used in
docs/decisions/t1_defaults.md. Everything is exact (no sampling) except where
noted, so the numbers are reproducible.

Three questions:
1. Motion noise -> failure floor. Even a robot that KNOWS its cell (the fully
   observable MDP) fails sometimes, because drift and overshoot cause collisions.
   For each maze we solve the MDP that maximises P(success) exactly and report
   1 - mean V(s) over uniform non-goal starts. No POMDP policy can beat this
   (full observability is an upper bound on the POMDP value), so every target
   failure rate d must lie above it.
2. Sensor range R -> information per look vs. observation-space size
   (R+2)^4, which drives the cost of the point-based POMDP solver.
3. Miss probability p_miss -> how much one look still identifies the cell,
   using the exact D6 observation model ("skip to next wall in range").
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

import maze_bank as mb
from motion_model import MotionParams, build_transition_model

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "docs" / "decisions" / "defaults_analysis.json"

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
RAY_STEP = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


# --------------------------------------------------------------------------
# 1. Motion-noise failure floor (fully observable MDP, max P(success))
# --------------------------------------------------------------------------

def success_values(wall_map: np.ndarray, goals, params: MotionParams, tol: float = 1e-12) -> np.ndarray:
    """V(s) = max P(reach a goal cell and DECLARE there without colliding)."""
    rows, cols = wall_map.shape[:2]
    T, _C = build_transition_model(wall_map, params)
    goal = np.zeros(rows * cols)
    for r, c in goals:
        goal[r * cols + c] = 1.0
    V = goal.copy()
    for _ in range(100_000):
        newV = np.maximum(goal, (T @ V).max(axis=0))
        if np.max(np.abs(newV - V)) < tol:
            return newV
        V = newV
    return V


def failure_floor(spec, params: MotionParams) -> float:
    V = success_values(spec.wall_map, spec.goal_cells, params)
    starts = [r * spec.cols + c for r in range(spec.rows) for c in range(spec.cols)
              if (r, c) not in set(spec.goal_cells)]
    return float(1.0 - V[starts].mean())


# --------------------------------------------------------------------------
# 2./3. Observation model under D5/D6
# --------------------------------------------------------------------------

def wall_lists(wall_map: np.ndarray, R: int) -> list[list[list[int]]]:
    """w[s][d] = sorted wall distances 0..R seen along ray d from cell s.

    Distance = number of cells the ray passes before the wall (0 = wall on the
    cell's own edge). Walls behind a missed wall are visible (D6); the outer
    border is the last wall a ray can see.
    """
    rows, cols = wall_map.shape[:2]
    out = []
    for r in range(rows):
        for c in range(cols):
            per_dir = []
            for d, (dr, dc) in RAY_STEP.items():
                dists, rr, cc, k = [], r, c, 0
                while k <= R:
                    if wall_map[rr, cc, d] == 0:
                        dists.append(k)
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    rr, cc, k = nr, nc, k + 1
                per_dir.append(dists)
            out.append(per_dir)
    return out


def observation_matrix(wall_map: np.ndarray, R: int, p_miss: float) -> np.ndarray:
    """Z[s, o], o in mixed radix base (R+2): digit R+1 means NONE."""
    base = R + 2
    w = wall_lists(wall_map, R)
    n = len(w)
    Z = np.zeros((n, base ** 4))
    for s in range(n):
        ray_probs = []
        for d in range(4):
            p = np.zeros(base)
            for j, delta in enumerate(w[s][d]):
                p[delta] += (p_miss ** j) * (1.0 - p_miss)
            p[R + 1] += p_miss ** len(w[s][d])
            ray_probs.append(p)
        z = ray_probs[0]
        for d in range(1, 4):
            z = np.outer(z, ray_probs[d]).ravel()
        Z[s] = z
    return Z


def one_look_stats(Z: np.ndarray, starts: list[int]) -> dict:
    """From a uniform belief over `starts`: expected posterior entropy H1 (nats),
    P(MAP cell is wrong) and the expected posterior mass on the true cell."""
    Zs = Z[starts]                          # (n, |O|)
    n = len(starts)
    p_o = Zs.sum(axis=0) / n                # P(o)
    keep = p_o > 0
    post = Zs[:, keep] / (Zs[:, keep].sum(axis=0, keepdims=True))   # b(s|o)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_o = -(np.where(post > 0, post * np.log(post), 0.0)).sum(axis=0)
    H1 = float((p_o[keep] * H_o).sum())
    p_map_right = float(Zs[:, keep].max(axis=0).sum() / n)
    true_mass = float((Zs[:, keep] * post).sum() / n)
    return {"H1": H1, "eff_cells": math.exp(H1), "p_map_wrong": 1.0 - p_map_right,
            "true_mass": true_mass}


# --------------------------------------------------------------------------
# 4. Statistics of a failure rate measured with n runs
# --------------------------------------------------------------------------

def wilson(f: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    center = (f + z * z / (2 * n)) / den
    half = z * math.sqrt(f * (1 - f) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


# --------------------------------------------------------------------------

def main() -> None:
    names = mb.list_mazes()
    specs = {name: mb.load_maze(name) for name in names}
    results: dict = {"mazes": names}

    # ---- 1. noise grid -> failure floor
    drifts = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    step_sets = {
        "exact (0,1,0)": (0.0, 1.0, 0.0),
        "stay-only (0.02,0.98,0)": (0.02, 0.98, 0.0),
        "low (0.02,0.975,0.005)": (0.02, 0.975, 0.005),
        "ProjectA (0.02,0.97,0.01)": None,     # resolved Gaussian form
        "hard (0.1,0.8,0.1)": (0.1, 0.8, 0.1),
    }
    pa = MotionParams.from_config(noisy=True).p_steps
    step_sets["ProjectA (0.02,0.97,0.01)"] = pa
    floor: dict = {}
    for (label, ps), pd in itertools.product(step_sets.items(), drifts):
        params = MotionParams(ps, pd, "collide")
        floor[f"{label} | drift={pd}"] = {name: failure_floor(specs[name], params) for name in names}
    results["floor"] = floor
    print("\n== 1. Failure floor with perfect knowledge (MDP), % — mean per size [max]")
    print(f"{'p_steps':28s} {'drift':>6s} | {'small':>13s} {'medium':>13s} {'large':>13s}")
    for key, per in floor.items():
        label, pd = key.split(" | drift=")
        row = []
        for size in mb.SIZES:
            vals = [100 * per[n] for n in names if n.startswith(size + "/")]
            row.append(f"{np.mean(vals):6.2f} [{max(vals):5.2f}]")
        print(f"{label:28s} {pd:>6s} | " + " ".join(row))

    # ---- 1b. the chosen presets, per maze
    from sim_config import load_config
    cfg_dir = Path(__file__).resolve().parent / "configs"
    presets = {"default": load_config(cfg_dir / "default.yaml").to_motion_params(),
               "stress": load_config(cfg_dir / "stress.yaml").to_motion_params()}
    chosen = {label: {name: failure_floor(specs[name], p) for name in names} for label, p in presets.items()}
    results["chosen_floor"] = chosen
    results["chosen_params"] = {k: {"p_steps": list(v.p_steps), "p_drift": v.p_drift} for k, v in presets.items()}
    print("\n== 1b. Failure floor of the chosen presets, % per maze")
    print(f"{'maze':22s} {'default':>8s} {'stress':>8s}")
    for name in names:
        print(f"{name:22s} {100 * chosen['default'][name]:8.3f} {100 * chosen['stress'][name]:8.3f}")

    # ---- 2. range R
    print("\n== 2. Range R (p_miss = 0): one look from a uniform start, mean per size")
    rinfo: dict = {}
    for R in (1, 2, 3, 4):
        rinfo[R] = {}
        for name in names:
            spec = specs[name]
            starts = [r * spec.cols + c for r in range(spec.rows) for c in range(spec.cols)
                      if (r, c) not in set(spec.goal_cells)]
            Z = observation_matrix(spec.wall_map, R, 0.0)
            st = one_look_stats(Z, starts)
            amb = mb.ambiguity(spec.wall_map, R=R)["limited"]
            rinfo[R][name] = {**st, "n_distinct": amb["n_distinct"], "n_shared": amb["n_shared"],
                              "largest_group": amb["largest_group"]}
    results["range"] = rinfo
    print(f"{'R':>2s} {'|O|':>5s} | " + " ".join(f"{s:>26s}" for s in mb.SIZES))
    for R in (1, 2, 3, 4):
        row = []
        for size in mb.SIZES:
            sel = [rinfo[R][n] for n in names if n.startswith(size + "/")]
            row.append(f"e^H1={np.mean([x['eff_cells'] for x in sel]):5.2f} "
                       f"shared={np.mean([x['n_shared'] for x in sel]):5.1f}")
        print(f"{R:2d} {(R + 2) ** 4:5d} | " + " ".join(f"{x:>26s}" for x in row))

    # ---- 3. p_miss at R = 2
    print("\n== 3. Miss probability at R = 2: one look from a uniform start, mean per size")
    minfo: dict = {}
    for pm in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
        minfo[pm] = {}
        for name in names:
            spec = specs[name]
            starts = [r * spec.cols + c for r in range(spec.rows) for c in range(spec.cols)
                      if (r, c) not in set(spec.goal_cells)]
            minfo[pm][name] = one_look_stats(observation_matrix(spec.wall_map, 2, pm), starts)
    results["p_miss"] = minfo
    print(f"{'p_miss':>6s} | " + " ".join(f"{s:>30s}" for s in mb.SIZES))
    for pm in minfo:
        row = []
        for size in mb.SIZES:
            sel = [minfo[pm][n] for n in names if n.startswith(size + "/")]
            row.append(f"e^H1={np.mean([x['eff_cells'] for x in sel]):5.2f} "
                       f"MAPwrong={100 * np.mean([x['p_map_wrong'] for x in sel]):5.1f}%")
        print(f"{pm:6.2f} | " + " ".join(f"{x:>30s}" for x in row))

    # ---- 4. measuring d
    print("\n== 4. 95% Wilson interval of a measured failure rate")
    stats = {}
    for d in (0.001, 0.01, 0.05, 0.1):
        stats[d] = {}
        for n in (500, 1000, 5000):
            lo, hi = wilson(d, n)
            stats[d][n] = (lo, hi)
        print(f"d={100 * d:5.1f}%  " + "  ".join(
            f"n={n}: [{100 * stats[d][n][0]:.2f}%, {100 * stats[d][n][1]:.2f}%]" for n in (500, 1000, 5000)))
    results["wilson"] = {str(d): {str(n): v for n, v in x.items()} for d, x in stats.items()}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
