"""Run Project B episodes headless in MuJoCo, through lidar_sim's runner (m1.3, m1.4).

Same flags as lidar_sim/runner.py, plus the MuJoCo config. Run from the mujuco folder:

    ..\\.venv\\Scripts\\python run_mujoco.py --policy projecta --set maze.name=small/m5_a --n-runs 100
    ..\\.venv\\Scripts\\python run_mujoco.py --policy projecta --mj-set noise.bad_return_on=true ...

The CSV has the same columns as the Python simulator's. Next to it,
``<out>.config.yaml`` holds the experiment config and ``<out>.mujoco.yaml`` the
MuJoCo config (controller, noise, mujoco version). The default output is
``results/mujoco_<policy>.csv`` so it is never mixed up with a Python-sim CSV.
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from pathlib import Path

import mujoco
import yaml

import sim  # noqa: F401  (puts lidar_sim on sys.path)
from sim.mj_config import load_mj_config
from sim.mj_grid_env import make_mujoco_env
from runner import build_parser, config_from_args, resolve_maze, run_and_write, sidecar_path, summarize
from sim_config import ConfigError

log = logging.getLogger("run_mujoco")


def main(argv=None) -> int:
    p = build_parser()
    p.description = __doc__.splitlines()[0]
    p.add_argument("--mj-config", default=None, help="MuJoCo YAML merged on top of configs/sim_default.yaml")
    p.add_argument("--mj-set", dest="mj_overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a MuJoCo config value (repeatable), e.g. --mj-set noise.friction_on=true")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = config_from_args(args)
        mj_cfg = load_mj_config(args.mj_config, args.mj_overrides)
    except (ConfigError, ValueError) as exc:
        log.error("config error: %s", exc)
        return 2
    if args.out is None:
        cfg = cfg.replace([f"run.out=results/mujoco_{cfg.policy.name}.csv"])
    maze = resolve_maze(cfg)

    t0 = time.perf_counter()
    factory = functools.partial(make_mujoco_env, mj_cfg=mj_cfg)
    out, rows = run_and_write(cfg, maze=maze, env_factory=factory)
    elapsed = time.perf_counter() - t0

    mj_side = Path(out).with_name(Path(out).stem + ".mujoco.yaml")
    mj_side.write_text(yaml.safe_dump({"mujoco": mujoco.__version__, **mj_cfg}, sort_keys=False), encoding="utf-8")

    s = summarize(rows)
    n = s["n"]
    log.info("outcomes: %s", s["outcomes"])
    log.info("failure rate %.4f, mean T %.2f (successes %.2f)", s["failure_rate"], s["mean_T"], s["mean_T_success"])
    log.info("%d episodes in %.1f s (%.2f s per episode)", n, elapsed, elapsed / max(n, 1))
    log.info("wrote %s, %s and %s", out, sidecar_path(out), mj_side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
