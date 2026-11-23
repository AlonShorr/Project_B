"""MJCF builder: no posts, each wall once, inconsistent maps rejected, every bank maze compiles."""

import mujoco
import numpy as np
import pytest

from sim.mjcf import build_mjcf, max_sensing_R, wall_geoms
import maze_bank
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP


def closed_edges(wm: np.ndarray) -> int:
    """Number of distinct closed edges (each interior edge counted once)."""
    n = int((wm[:, :, WALL_UP] == 0).sum() + (wm[:, :, WALL_LEFT] == 0).sum())
    n += int((wm[-1, :, WALL_DOWN] == 0).sum() + (wm[:, -1, WALL_RIGHT] == 0).sum())
    return n


@pytest.mark.parametrize("name", maze_bank.list_mazes())
def test_each_wall_once_no_posts_and_compiles(name):
    wm = maze_bank.load_maze(name).wall_map
    geoms = wall_geoms(wm)
    names = [g.split('"')[1] for g in geoms]
    assert len(names) == len(set(names))
    assert not any(n.startswith("post_") for n in names)
    assert len(geoms) == closed_edges(wm)
    model = mujoco.MjModel.from_xml_string(build_mjcf(wm, (0, 0), R=2))
    assert model.ngeom == len(geoms) + 2          # + floor + robot
    assert model.nsensor == 4


def test_smoke_maze_geom_count():
    wm = maze_bank.open_wall_map(4, 4)
    maze_bank.set_edge(wm, 1, 1, WALL_RIGHT, 0)
    model = mujoco.MjModel.from_xml_string(build_mjcf(wm, (1, 1), R=3))
    assert model.ngeom == 19                       # 17 walls + floor + robot (the guide's 44 had 25 posts)


def test_one_sided_wall_is_rejected():
    wm = maze_bank.open_wall_map(3, 3)
    wm[1, 1, WALL_RIGHT] = 0                       # (1,2) still says LEFT is open
    with pytest.raises(ValueError, match="inconsistent"):
        wall_geoms(wm)


def test_R_limited_by_sensor_range():
    assert max_sensing_R() == 6                    # VL53L4CD 1.3 m
    wm = maze_bank.open_wall_map(3, 3)
    build_mjcf(wm, (0, 0), R=6)
    with pytest.raises(ValueError, match="sensor range"):
        build_mjcf(wm, (0, 0), R=7)


def test_dirichlet_summary_has_finite_intervals():
    from sim.calibrate import dirichlet_summary
    only = dirichlet_summary({"collision": 4})
    assert only == {"collision": {"p": 1.0, "ci": [1.0, 1.0], "n": 4}}
    mixed = dirichlet_summary({"moved_+0_+1": 248})
    assert set(mixed) == {"collision", "moved_+0_+1"}
    assert all(np.isfinite(v["ci"]).all() and 0 < v["ci"][1] <= 1 for v in mixed.values())
    assert mixed["collision"]["ci"][1] < 0.02     # never seen in 248: honest upper bound only


def test_calibration_context_blocks_illegal_diagonals():
    from sim.calibrate import context
    wm = maze_bank.open_wall_map(4, 4)
    maze_bank.set_edge(wm, 1, 1, WALL_RIGHT, 0)     # wall on the edge between (1,1) and (1,2)
    assert context(wm, (3, 3), (-1, -1), 1) == "open"
    assert context(wm, (2, 1), (-1, 1), 1) == "blocked"   # NE from (2,1): corner touches that wall
    assert context(wm, (2, 2), (-1, -1), 1) == "blocked"  # NW from (2,2): the same corner
    assert context(wm, (1, 1), (0, 1), 1) == "blocked"
