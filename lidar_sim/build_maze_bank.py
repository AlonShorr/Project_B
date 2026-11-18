"""Regenerate the maze bank (roadmap task t6) deterministically.

Run from this directory:
    python build_maze_bank.py            # rewrite mazes/<size>/*.npz|.txt and mazes/index.yaml
    python build_maze_bank.py --sweep    # only print the seed sweep used for selection

Generated mazes: for every recipe ``(name, loop_fraction, quantile)`` the
script generates ``SWEEP_SEEDS`` mazes with fixed seeds, sorts them by the
R=2 expected posterior entropy ``H1`` (ties by seed), and keeps the one at that
quantile. The suffixes ``a``/``b``/``c`` are the recipe order (loop fraction
high to low, H1 quantile low to high), i.e. the *intended* easy-to-hard order.
Everything is seeded, so the output is identical on every run.

Difficulty labels in index.yaml are *measured*, not taken from the suffix:
within each size class, every maze (generated or predecessor) is ranked by its
R=2 ``H1`` rounded to 6 decimals. A maze's position is its mid-rank among the
``n`` mazes of the size (tied H1 values share the average of their ranks, so
they always get the same label), divided by ``n - 1``: below 1/3 is ``easy``,
below 2/3 ``medium``, otherwise ``hard``. The labels are therefore relative to
the other mazes of the same size, not absolute.

Predecessor mazes: copied (MIT licence, see mazes/predecessor/LICENSE) from the
predecessor team's micromouse-py-sim and converted with
``maze_bank.maze_to_wall_map``. Their goals are replaced by the bank's center
goal so that every maze of a size uses the same goal convention.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

import maze_bank as mb

SIZE_SHAPES = mb.SIZE_SHAPES
SWEEP_SEEDS = 100
R_VALUES = (1, 2, 3)
R_HEADLINE = 2

# (name, loop_fraction, quantile of H1 over the seed sweep)
RECIPES = {
    "small": (("m5_a", 0.35, 0.10), ("m5_b", 0.10, 0.50), ("m5_c", 0.00, 0.95)),
    "medium": (("m10_a", 0.35, 0.10), ("m10_b", 0.10, 0.50), ("m10_c", 0.00, 0.95)),
    "large": (("m16_a", 0.35, 0.10), ("m16_b", 0.10, 0.50), ("m16_c", 0.00, 0.95)),
}
SEED_BASE = {"small": 5000, "medium": 10000, "large": 16000}

PREDECESSOR_DIR = mb.MAZE_DIR / "predecessor"
PREDECESSOR_ORIGIN = "micromouse-py-sim by Roy Abudi & Huly (predecessor team), MIT licence"
# name -> (size, file, original preset start, original preset goals) from presets.json
PREDECESSOR = {
    "simple": ("small", "simple.maze", (0, 0), ((1, 2),)),
    "longer": ("small", "longer.maze", (4, 0), ((1, 3),)),
    "japan_2017": ("large", "japan_2017.maze", (15, 0), ((7, 7), (7, 8), (8, 7), (8, 8))),
    "semifinal_2009": ("large", "semifinal_2009.maze", (15, 0), ((7, 7), (7, 8), (8, 7), (8, 8))),
}


def _seeds(size: str) -> range:
    base = SEED_BASE[size]
    return range(base, base + SWEEP_SEEDS)


def select_seed(size: str, loop_fraction: float, quantile: float) -> tuple[int, list[tuple[float, int]]]:
    """Seed at the given H1 quantile among the fixed sweep. Returns (seed, sorted sweep)."""
    rows, cols = SIZE_SHAPES[size]
    scored = []
    for seed in _seeds(size):
        w = mb.generate_maze(rows, cols, seed, loop_fraction)
        scored.append((round(mb.ambiguity(w, R_HEADLINE)["limited"]["H1"], 9), seed))
    scored.sort()
    idx = min(len(scored) - 1, int(round(quantile * (len(scored) - 1))))
    return scored[idx][1], scored


def _rounded(stats: dict) -> dict:
    return {k: (round(float(v), 6) if isinstance(v, float) else int(v)) for k, v in stats.items()}


def ambiguity_entry(wall_map: np.ndarray) -> dict:
    entry = {"unbounded": _rounded(mb.ambiguity(wall_map, R_HEADLINE)["unbounded"])}
    for R in R_VALUES:
        entry[f"R{R}"] = _rounded(mb.ambiguity(wall_map, R)["limited"])
    return entry


def build_specs() -> list[mb.MazeSpec]:
    specs = []
    for size, recipes in RECIPES.items():
        rows, cols = SIZE_SHAPES[size]
        for name, loop_fraction, quantile in recipes:
            seed, _ = select_seed(size, loop_fraction, quantile)
            wall_map = mb.generate_maze(rows, cols, seed, loop_fraction)
            source = {
                "kind": "generator",
                "seed": seed,
                "loop_fraction": loop_fraction,
                "dfs_start": [0, 0],
                "selection": (
                    f"seed at H1 quantile {quantile} (R={R_HEADLINE}) among seeds "
                    f"{_seeds(size).start}..{_seeds(size).stop - 1}"
                ),
            }
            specs.append(mb.MazeSpec(f"{size}/{name}", wall_map, mb.center_goal(rows, cols), rows, cols, {"source": source}))

    for name, (size, fname, start, goals) in PREDECESSOR.items():
        path = PREDECESSOR_DIR / fname
        wall_map = mb.maze_to_wall_map(path)
        rows, cols = wall_map.shape[:2]
        if (rows, cols) != SIZE_SHAPES[size]:
            raise ValueError(f"{fname} is {rows}x{cols}, expected {SIZE_SHAPES[size]}")
        problems = mb.wall_map_problems(wall_map)
        if problems:
            raise ValueError(f"{fname} does not convert cleanly: {problems}")
        source = {
            "kind": "predecessor",
            "file": f"mazes/predecessor/{fname}",
            "origin": PREDECESSOR_ORIGIN,
            "original_start": list(start),
            "original_goals": [list(g) for g in goals],
        }
        specs.append(mb.MazeSpec(f"{size}/{name}", wall_map, mb.center_goal(rows, cols), rows, cols, {"source": source}))
    return specs


def _difficulty_labels(entries: dict[str, dict]) -> dict[str, str]:
    """easy/medium/hard by mid-rank of R=2 H1 within each size; ties share a label."""
    labels = {}
    for size in SIZE_SHAPES:
        h1 = {n: round(e["ambiguity"][f"R{R_HEADLINE}"]["H1"], 6) for n, e in entries.items() if e["size"] == size}
        n_mazes = len(h1)
        for name, value in h1.items():
            lower = sum(1 for v in h1.values() if v < value)
            equal = sum(1 for v in h1.values() if v == value)
            mid_rank = lower + (equal - 1) / 2  # 0-based average rank of the tie group
            frac = mid_rank / max(1, n_mazes - 1)
            labels[name] = "easy" if frac < 1 / 3 else ("medium" if frac < 2 / 3 else "hard")
    return labels


def build(root: Path | None = None) -> dict:
    root = Path(root or mb.MAZE_DIR)
    for size in SIZE_SHAPES:  # start clean so removed recipes leave no stale files
        for old in list((root / size).glob("*.npz")) + list((root / size).glob("*.txt")):
            old.unlink()

    entries = {}
    for spec in build_specs():
        mb.save_maze(spec, root=root)
        entries[spec.name] = {
            "size": spec.size,
            "rows": spec.rows,
            "cols": spec.cols,
            "goal_cells": [list(g) for g in spec.goal_cells],
            "source": spec.meta["source"],
            "ambiguity": ambiguity_entry(spec.wall_map),
        }
    for name, label in _difficulty_labels(entries).items():
        entries[name]["difficulty"] = label

    index = {
        "version": 1,
        "generated_by": "lidar_sim/build_maze_bank.py",
        "mazes": dict(sorted(entries.items())),
    }
    (root / "index.yaml").write_text(
        INDEX_HEADER + yaml.safe_dump(index, sort_keys=False, default_flow_style=None, width=120), encoding="utf-8"
    )
    return index


INDEX_HEADER = """\
# Maze bank index (roadmap t6). GENERATED by build_maze_bank.py - do not edit by hand.
# Load a maze with maze_bank.load_maze("<size>/<name>").
#
# wall_map   : (rows, cols, 4) uint8, channels UP, DOWN, LEFT, RIGHT; 0 = wall, 1 = open; row 0 = north.
# goal_cells : center cell for odd sizes, 2x2 center block for even sizes.
# signature  : 4 absolute cardinal rays; d = number of open cells to the first wall
#              (d = 0: wall on the cell's own edge).
#              unbounded = Project A score_map signature (d itself, any range);
#              R<k>      = noiseless range-limited reading: d if d <= k else NONE.
# stats      : n_shared     = cells whose signature is shared with >= 1 other cell (headline number)
#              n_distinct   = number of distinct signatures; amb = 1 - n_distinct / n_cells
#              largest_group = most cells sharing one signature
#              H1           = sum_g |g|/N * ln|g| (nats): expected posterior entropy after one
#                             noiseless look from a uniform belief (0 = one look localizes; ln N = useless)
# difficulty : easy / medium / hard by mid-rank of R2.H1 within the size class (relative; ties share
#              a label). Name suffixes a/b/c are the generator recipe order, not the label.
"""


def print_table(index: dict) -> None:
    hdr = f"{'maze':22s} {'diff':6s} {'unb shared':>10s} " + " ".join(
        f"{'R%d shared/distinct/H1' % R:>24s}" for R in R_VALUES
    )
    print(hdr)
    for name, e in index["mazes"].items():
        a = e["ambiguity"]
        cols = " ".join(
            f"{a[f'R{R}']['n_shared']:>9d}/{a[f'R{R}']['n_distinct']:>3d}/{a[f'R{R}']['H1']:6.3f}    " for R in R_VALUES
        )
        n = e["rows"] * e["cols"]
        print(f"{name:22s} {e['difficulty']:6s} {a['unbounded']['n_shared']:>6d}/{n:<3d} {cols}  ln N={math.log(n):.2f}")


def sweep() -> None:
    for size, recipes in RECIPES.items():
        for name, lf, q in recipes:
            seed, scored = select_seed(size, lf, q)
            h = [s for s, _ in scored]
            print(f"{size}/{name}: lf={lf} q={q} -> seed {seed}; H1 min/med/max {min(h):.3f}/{np.median(h):.3f}/{max(h):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="print the selection sweep only, write nothing")
    args = parser.parse_args()
    if args.sweep:
        sweep()
        return
    index = build()
    print_table(index)
    print(f"wrote {len(index['mazes'])} mazes and {mb.INDEX_PATH}")


if __name__ == "__main__":
    main()
