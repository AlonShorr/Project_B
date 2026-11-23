"""m3 calibration entry point (setup guide §7). Run from the mujuco folder:

    ..\\.venv\\Scripts\\python run_calibration.py                 # placeholder noise -> build/
    ..\\.venv\\Scripts\\python run_calibration.py --n 1000 --out ..\\lidar_sim\\configs\\calibrated.yaml

The NoiseConfig defaults are placeholders until m2.4, so the default output goes
to build/ (git-ignored). Write to lidar_sim/configs/ only with measured parameters,
and commit that YAML together with the code version that produced it.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from sim import BUILD_DIR
from sim.calibrate import CARDINAL, DIAGONAL, dirichlet_summary, p_miss, run_batch, write_yaml
from sim.noise import NoiseConfig
from sim.noisy_env import NoisyMazeSim
from config import WALL_RIGHT
from maze_bank import open_wall_map, set_edge


def calibration_maze() -> np.ndarray:
    """8x8 open box plus a few inner walls so every context (open, side_wall, blocked) occurs."""
    wm = open_wall_map(8, 8)
    for r, c in [(2, 2), (2, 5), (5, 3), (4, 6)]:
        set_edge(wm, r, c, WALL_RIGHT, 0)
    return wm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=1000, help="commands per action type (split over its directions)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--R", type=int, default=None, help="sensing range (default: config.SENSING_RANGE_R)")
    p.add_argument("--out", default=str(BUILD_DIR / "calibrated.yaml"))
    args = p.parse_args()

    wm = calibration_maze()
    cfg = NoiseConfig.all_on()          # placeholder params until m2.4
    sim = NoisyMazeSim(wm, start=(1, 1), R=args.R, noise=cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)

    t0 = time.perf_counter()
    motion: dict = {}
    for label, actions in [("cardinal", CARDINAL), ("diagonal", DIAGONAL)]:
        counts = run_batch(sim, wm, actions, k=1, n=args.n // len(actions), rng=rng)
        for (name, ctx), cnt in sorted(counts.items()):
            motion.setdefault(label, {}).setdefault(name, {})[ctx] = dirichlet_summary(dict(cnt))
    miss = p_miss(sim, wm, n=500, rng=rng)

    BUILD_DIR.mkdir(exist_ok=True)
    write_yaml(args.out, motion, miss,
               meta={"mujoco": mujoco.__version__, "seed": args.seed, "n_per_type": args.n,
                     "R": sim.R, "noise": cfg.__dict__})
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f} s; "
          f"p_miss (wrong reading) = {miss['p']:.3f} CI {np.round(miss['ci'], 3).tolist()} over {miss['n']} rays")


if __name__ == "__main__":
    main()
