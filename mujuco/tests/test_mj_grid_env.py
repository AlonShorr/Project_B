"""m1.3: MujocoGridEnv has GridEnv's interface, and the runner's policies run on it unchanged."""

import functools

import numpy as np
import pytest

from sim.mj_config import load_mj_config
from sim.mj_grid_env import MujocoGridEnv, make_mujoco_env
import maze_bank
from grid_env import GridEnv
from motion_model import Action, Outcome
from policies import EnvInfo
from runner import LEAD_COLUMNS, LoadedMaze, run_batch, summarize
from sim_config import load_config

CFG = load_config()
SMALL = LoadedMaze.from_spec(maze_bank.load_maze("small/m5_a"))
MJ = load_mj_config()


def _envs(maze=SMALL, cfg=CFG):
    kw = dict(params=cfg.to_motion_params(), max_steps=cfg.episode.max_steps, sensing=cfg.to_sensing_params())
    return GridEnv(maze.wall_map, maze.goal_cells, **kw), MujocoGridEnv(maze.wall_map, maze.goal_cells, **kw)


def test_same_public_interface_as_grid_env():
    grid, mj = _envs()
    for name in ("wall_map", "rows", "cols", "goal_cells", "params", "max_steps", "sensing",
                 "reset", "step", "done", "log", "start_cells"):
        assert hasattr(mj, name), name
    a, b = EnvInfo.from_env(grid), EnvInfo.from_env(mj)
    assert (a.rows, a.cols, a.goal_cells, a.motion, a.max_steps, a.sensing) == \
           (b.rows, b.cols, b.goal_cells, b.motion, b.max_steps, b.sensing)


def test_same_start_cell_per_seed_as_grid_env():
    grid, mj = _envs()
    for seed in range(30):
        assert grid.reset(seed).info["cell"] == mj.reset(seed).info["cell"]


def test_noiseless_first_observation_matches_grid_env_without_misses():
    cfg = CFG.replace(["sensing.p_miss=0.0"])
    grid, mj = _envs(cfg=cfg)
    for seed in range(20):
        assert grid.reset(seed).obs == mj.reset(seed).obs


def test_episode_outcomes_and_log():
    _, mj = _envs()
    res = mj.reset(3)
    start = res.info["cell"]
    # walk into the border: N from row 0, or keep going N until a collision
    for _ in range(10):
        res = mj.step(Action.N)
        if mj.done:
            break
    assert mj.log.outcome is Outcome.COLLISION and mj.log.collision is not None
    assert mj.log.trajectory[0] == start and mj.log.final_cell == res.info["cell"]
    with pytest.raises(RuntimeError):
        mj.step(Action.N)
    mj.reset(3)
    res = mj.step(Action.DECLARE)
    assert res.outcome is Outcome.WRONG_DECLARATION and res.obs is None   # starts are never goals


def test_timeout_after_max_steps():
    wm = maze_bank.open_wall_map(5, 5)
    mj = MujocoGridEnv(wm, [(4, 4)], max_steps=3, sensing=CFG.to_sensing_params())
    mj.reset(0, start=(2, 2))
    outcomes = [mj.step(a).outcome for a in (Action.E, Action.W, Action.NE)]
    assert outcomes == [Outcome.RUNNING, Outcome.RUNNING, Outcome.TIMEOUT]
    assert mj.log.trajectory == [(2, 2), (2, 3), (2, 2), (1, 3)]
    assert [m.k for m in mj.log.moves] == [1, 1, 1] and mj.log.moves[2].direction is Action.NE


@pytest.mark.parametrize("robot", ["mecanum", "point"])
@pytest.mark.parametrize("policy", ["oracle_bfs", "projecta"])
def test_runner_runs_policies_unchanged(policy, robot):
    cfg = CFG.replace([f"policy.name={policy}", "run.n_runs=4", "run.seed0=0"])
    factory = functools.partial(make_mujoco_env, mj_cfg=load_mj_config(overrides=[f"robot={robot}"]))
    rows = run_batch(cfg, SMALL, env_factory=factory)
    assert [r["seed"] for r in rows] == [0, 1, 2, 3]
    assert all(set(LEAD_COLUMNS) <= set(r) for r in rows)
    assert all(r["outcome"] != Outcome.RUNNING.value for r in rows)
    if policy == "oracle_bfs":                       # noise off: the oracle always arrives
        assert summarize(rows)["outcomes"]["success"] == 4


@pytest.mark.parametrize("overrides", [
    [f"mecanum.noise.{f}=true" for f in ("radius_on", "strafe_on", "slip_on", "tof_noise_on", "bad_return_on")],
    ["robot=point"] + [f"noise.{f}=true" for f in
                       ("friction_on", "vel_noise_on", "stop_jitter_on", "range_noise_on", "bad_return_on")],
])
def test_runner_is_reproducible_with_noise_on(overrides):
    mj_cfg = load_mj_config(overrides=overrides)
    cfg = CFG.replace(["policy.name=projecta", "run.n_runs=2", "run.seed0=5"])
    factory = functools.partial(make_mujoco_env, mj_cfg=mj_cfg)
    a = run_batch(cfg, SMALL, env_factory=factory)
    b = run_batch(cfg, SMALL, env_factory=factory)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_time_ms"} for r in rows]
    assert strip(a) == strip(b)


def test_mj_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        load_mj_config(overrides=["noise.dropout_on=true"])
    with pytest.raises(ValueError, match="unknown"):
        load_mj_config(overrides=["mecanum.tof.fov=20"])
    with pytest.raises(ValueError, match="robot"):
        load_mj_config(overrides=["robot=tank"])
    cfg = load_mj_config(overrides=["controller.v_max=0.2", "mecanum.tof.fov_deg=27"])
    assert cfg["controller"]["v_max"] == 0.2 and cfg["mecanum"]["tof"]["fov_deg"] == 27
    assert load_mj_config()["robot"] == "mecanum"
