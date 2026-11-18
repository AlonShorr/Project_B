"""Headless batch runner for Project B episodes (t5).

    python runner.py --config configs/default.yaml --n-runs 1000 --seed0 0 \\
        --out results/oracle.csv --workers 4 --set motion.p_drift=0.2

For each seed in ``seed0 .. seed0 + n_runs - 1`` one episode runs on the
configured maze and one CSV row is written. Results are sorted by seed, so the
file is the same for any worker count (apart from ``wall_time_ms``). A
``<out stem>.config.yaml`` sidecar next to the CSV holds the resolved config.

Randomness: the environment uses ``default_rng(seed)`` (the same start cell as
``GridEnv.reset(seed)`` everywhere else), and the policy uses
``default_rng(SeedSequence(seed).spawn(1)[0])``, an independent stream. Changing
the policy therefore never changes the start cells or the motion noise drawn
for the same action sequence (common random numbers for paired comparisons).

This module must not import pygame (directly or through other modules).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from grid_env import GridEnv
from motion_model import Action, Outcome
from policies import EnvInfo, Policy, make_policy
from sim_config import DEFAULT_CONFIG_PATH, ConfigError, SimConfig, load_config

log = logging.getLogger("runner")

Cell = tuple[int, int]

# Columns before the flattened ``section.key`` parameters and after them.
LEAD_COLUMNS = (
    "run_id", "seed", "maze", "maze_hash", "policy", "policy_params_json",
    "T", "outcome", "failed", "n_moves",
    "steps_to_loc", "final_P_goal", "n_fallbacks",
    "start_r", "start_c", "final_r", "final_c",
    "collision_action", "collision_direction", "collision_k",
)
TAIL_COLUMNS = ("wall_time_ms", "config_hash")
POLICY_STAT_COLUMNS = ("steps_to_loc", "final_P_goal", "n_fallbacks")


# --------------------------------------------------------------------------
# Maze
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedMaze:
    """A maze ready to run: picklable, with the goal override applied."""

    name: str
    wall_map: np.ndarray
    goal_cells: tuple[Cell, ...]

    @property
    def rows(self) -> int:
        return int(self.wall_map.shape[0])

    @property
    def cols(self) -> int:
        return int(self.wall_map.shape[1])

    @property
    def maze_hash(self) -> str:
        h = hashlib.sha256()
        h.update(np.asarray(self.wall_map.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(self.wall_map != 0, dtype=np.uint8).tobytes())
        h.update(json.dumps(sorted(self.goal_cells)).encode())
        return h.hexdigest()[:16]

    @classmethod
    def from_spec(cls, spec: Any, goal: Iterable[Cell] | None = None, name: str | None = None) -> "LoadedMaze":
        """Build from anything with ``wall_map`` and ``goal_cells`` (e.g. maze_bank.MazeSpec)."""
        wall_map = np.asarray(spec.wall_map)
        if wall_map.ndim != 3 or wall_map.shape[2] != 4:
            raise ValueError(f"wall_map must have shape (rows, cols, 4), got {wall_map.shape}")
        goals = goal if goal is not None else spec.goal_cells
        goals = tuple(sorted({(int(r), int(c)) for r, c in goals}))
        rows, cols = wall_map.shape[:2]
        for r, c in goals:
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"goal cell {(r, c)} is outside the {rows}x{cols} maze")
        return cls(name=name or str(getattr(spec, "name", "")), wall_map=wall_map, goal_cells=goals)


def resolve_maze(cfg: SimConfig) -> LoadedMaze:
    """Load ``cfg.maze.name`` from the maze bank and apply ``cfg.maze.goal``."""
    import maze_bank  # lazy: only needed when the maze is not passed in directly

    spec = maze_bank.load_maze(cfg.maze.name)
    return LoadedMaze.from_spec(spec, goal=cfg.maze.goal, name=getattr(spec, "name", None) or cfg.maze.name)


def effective_config(cfg: SimConfig, maze: LoadedMaze) -> SimConfig:
    """The config as actually run: ``maze.name`` is the name of the maze used.

    Differs from ``cfg`` only when a maze is passed in directly (tests,
    scripts) or the bank canonicalised a bare name. The goal cells actually
    used are part of ``maze.maze_hash``.
    """
    if cfg.maze.name == maze.name:
        return cfg
    data = cfg.to_dict()
    data["maze"]["name"] = maze.name
    return SimConfig.from_dict(data)


def experiment_hash(cfg: SimConfig, maze: LoadedMaze) -> str:
    """The CSV ``config_hash``: resolved config + the content of the maze run."""
    return effective_config(cfg, maze).config_hash(maze.maze_hash)


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------

class EpisodeRunner:
    """Holds one env and one policy; runs episodes by seed. Built once per worker."""

    def __init__(self, cfg: SimConfig, maze: LoadedMaze):
        cfg = effective_config(cfg, maze)
        self.cfg = cfg
        self.maze = maze
        self.env = GridEnv(
            maze.wall_map,
            maze.goal_cells,
            params=cfg.to_motion_params(),
            max_steps=cfg.episode.max_steps,
        )
        self.env_info = EnvInfo.from_env(self.env)
        self.policy: Policy = make_policy(cfg.policy.name, cfg.policy.selected_params)
        self.config_hash = cfg.config_hash(maze.maze_hash)
        self.static = {
            "maze": maze.name,
            "maze_hash": maze.maze_hash,
            "policy": cfg.policy.name,
            "policy_params_json": json.dumps(self.policy.params, sort_keys=True, separators=(",", ":")),
            **cfg.flat_params(),
            "config_hash": self.config_hash,
        }

    def run(self, seed: int) -> dict[str, Any]:
        env, policy = self.env, self.policy
        t0 = time.perf_counter()
        res = env.reset(seed=seed)
        policy.reset(self.env_info, policy_rng(seed))
        privileged = policy.privileged
        obs, info, last = res.obs, res.info, None
        while not env.done:
            if not privileged:
                info = {"t": env.log.steps, "last_action": last}
            last = policy.act(obs, info)
            res = env.step(last)
            obs, info = res.obs, res.info
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return self._row(seed, env.log, policy.episode_stats(), wall_ms)

    def _row(self, seed, ep, stats, wall_ms) -> dict[str, Any]:
        declared = ep.outcome in (Outcome.SUCCESS, Outcome.WRONG_DECLARATION)
        n_moves = len(ep.moves)
        T = n_moves * self.cfg.episode.step_cost + (self.cfg.episode.declare_cost if declared else 0.0)
        base = ep.as_row()
        row = {
            "run_id": f"{self.config_hash}-{seed}",
            "seed": seed,
            "T": T,
            "outcome": ep.outcome.value,
            "failed": int(ep.failed),
            "n_moves": n_moves,
            **{k: stats.get(k) for k in POLICY_STAT_COLUMNS},
            **{k: base[k] for k in ("start_r", "start_c", "final_r", "final_c",
                                    "collision_action", "collision_direction", "collision_k")},
            "wall_time_ms": round(wall_ms, 3),
        }
        row.update(self.static)
        return row


def policy_rng(seed: int) -> np.random.Generator:
    """The policy's generator: independent of the env's ``default_rng(seed)``."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


# Per-process runner for the multiprocessing pool.
_WORKER: EpisodeRunner | None = None


def _init_worker(cfg_dict: dict[str, Any], maze: LoadedMaze) -> None:
    global _WORKER
    _WORKER = EpisodeRunner(SimConfig.from_dict(cfg_dict), maze)


def _run_seed(seed: int) -> dict[str, Any]:
    return _WORKER.run(seed)


def run_batch(
    cfg: SimConfig,
    maze: LoadedMaze | None = None,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run ``cfg.run.n_runs`` episodes and return the rows sorted by seed."""
    maze = maze if maze is not None else resolve_maze(cfg)
    cfg = effective_config(cfg, maze)
    workers = cfg.run.workers if workers is None else int(workers)
    seeds = range(cfg.run.seed0, cfg.run.seed0 + cfg.run.n_runs)
    n = len(seeds)
    every = max(1, n // 10)
    log.info(
        "running %d episodes: maze=%s (%dx%d) policy=%s workers=%d config_hash=%s",
        n, maze.name, maze.rows, maze.cols, cfg.policy.name, workers, cfg.config_hash(maze.maze_hash),
    )
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        runner = EpisodeRunner(cfg, maze)
        for i, seed in enumerate(seeds, 1):
            rows.append(runner.run(seed))
            if i % every == 0:
                log.info("%d/%d episodes, %.1f s", i, n, time.perf_counter() - t0)
    else:
        chunksize = max(1, n // (workers * 8))
        with mp.Pool(workers, initializer=_init_worker, initargs=(cfg.to_dict(), maze)) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_seed, seeds, chunksize=chunksize), 1):
                rows.append(row)
                if i % every == 0:
                    log.info("%d/%d episodes, %.1f s", i, n, time.perf_counter() - t0)
    rows.sort(key=lambda r: r["seed"])
    log.info("finished %d episodes in %.2f s", n, time.perf_counter() - t0)
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def csv_columns(cfg: SimConfig) -> list[str]:
    return [*LEAD_COLUMNS, *cfg.flat_params(), *TAIL_COLUMNS]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path, columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})
    return path


def sidecar_path(out: str | Path) -> Path:
    out = Path(out)
    return out.with_name(out.stem + ".config.yaml")


def write_sidecar(cfg: SimConfig, maze: LoadedMaze, out: str | Path) -> Path:
    """Write the resolved config next to the CSV.

    The body is a loadable config (``load_config(sidecar)`` reproduces the run
    with the same ``config_hash`` when the same maze is loaded). The header
    comments record the maze actually run: name, size, goal cells and hash.
    """
    cfg = effective_config(cfg, maze)
    path = sidecar_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    goals = json.dumps([list(g) for g in maze.goal_cells])
    header = "".join(f"# {line}\n" for line in (
        f"Resolved config for {Path(out).name}",
        f"config_hash: {cfg.config_hash(maze.maze_hash)}   (config + maze content)",
        f"maze_name: {maze.name}",
        f"maze_size: {maze.rows}x{maze.cols}",
        f"maze_goal_cells: {goals}",
        f"maze_hash: {maze.maze_hash}",
        f"motion.p_steps resolved: {list(cfg.motion.resolved_p_steps())}",
    ))
    path.write_text(header + cfg.to_yaml(), encoding="utf-8")
    return path


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts: dict[str, int] = {o.value: 0 for o in Outcome if o is not Outcome.RUNNING}
    for row in rows:
        counts[row["outcome"]] += 1
    succ = [row["T"] for row in rows if row["outcome"] == Outcome.SUCCESS.value]
    return {
        "n": n,
        "outcomes": counts,
        "failure_rate": (n - counts[Outcome.SUCCESS.value]) / n if n else float("nan"),
        "mean_T_success": float(np.mean(succ)) if succ else float("nan"),
        "mean_T": float(np.mean([row["T"] for row in rows])) if rows else float("nan"),
    }


def run_and_write(cfg: SimConfig, maze: LoadedMaze | None = None, workers: int | None = None) -> tuple[Path, list[dict]]:
    maze = maze if maze is not None else resolve_maze(cfg)
    rows = run_batch(cfg, maze, workers=workers)
    out = write_csv(rows, cfg.run.out, csv_columns(cfg))
    write_sidecar(cfg, maze, cfg.run.out)
    return out, rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Project B episodes headless and write one CSV row per episode.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                   help="YAML config, merged on top of configs/default.yaml (default: %(default)s)")
    p.add_argument("--n-runs", type=int, help="override run.n_runs")
    p.add_argument("--seed0", type=int, help="override run.seed0")
    p.add_argument("--out", help="override run.out (CSV path)")
    p.add_argument("--workers", type=int, help="override run.workers")
    p.add_argument("--policy", help="override policy.name")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any config value (repeatable), e.g. --set motion.p_drift=0.2")
    p.add_argument("--log-level", default="INFO", help="logging level (default: %(default)s)")
    return p


def config_from_args(args: argparse.Namespace) -> SimConfig:
    """Config file, then ``--set`` overrides, then the dedicated flags."""
    data = load_config(args.config, args.overrides).to_dict()
    for flag, (section, key) in {
        "n_runs": ("run", "n_runs"), "seed0": ("run", "seed0"), "out": ("run", "out"),
        "workers": ("run", "workers"), "policy": ("policy", "name"),
    }.items():
        value = getattr(args, flag)
        if value is not None:
            data[section][key] = value
    return SimConfig.from_dict(data)


def main(argv: Sequence[str] | None = None, maze: LoadedMaze | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = config_from_args(args)
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return 2
    if maze is None:
        try:
            maze = resolve_maze(cfg)
        except (ImportError, KeyError, OSError, ValueError) as exc:
            log.error("cannot load maze %r: %s", cfg.maze.name, exc)
            return 2
    out, rows = run_and_write(cfg, maze=maze)
    s = summarize(rows)
    log.info("outcomes: %s", s["outcomes"])
    n = sum(s["outcomes"].values())
    log.info(
        "failure rate %.4f (%d of %d), mean T %.2f (successes %.2f)",
        s["failure_rate"], round(s["failure_rate"] * n), n, s["mean_T"], s["mean_T_success"],
    )
    log.info("wrote %s and %s", out, sidecar_path(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
