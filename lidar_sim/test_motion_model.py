"""Tests for the Project B motion model (roadmap task t3).

Run from this directory:
    python -m pytest test_motion_model.py -q
"""

import math

import numpy as np
import pytest

import config
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from grid_env import GridEnv, outcome_counts
from localization import predict_forward_8dir
from map_generator import get_maps
from motion_model import (
    DIR_DELTAS,
    MOVE_ACTIONS,
    Action,
    MotionParams,
    Outcome,
    build_transition_model,
    heading_to_action,
    can_step,
    cell_index,
    index_to_cell,
    move_distribution,
    open_wall_map,
    resolve_move,
    sample_move,
    set_wall,
)
from sim_robot import Robot

PROJECT_A = MotionParams.from_config(noisy=True)
HARD = MotionParams.hard()
DET = MotionParams.deterministic()
STOP_HARD = MotionParams(HARD.p_steps, HARD.p_drift, "stop")
ALL_PARAMS = [PROJECT_A, HARD, DET, STOP_HARD]
PARAM_IDS = ["projectA", "hard", "deterministic", "stop_hard"]


def random_maze(rows, cols, seed, density=0.3):
    rng = np.random.default_rng(seed)
    wall_map = open_wall_map(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and rng.random() < density:
                set_wall(wall_map, r, c, WALL_RIGHT)
            if r + 1 < rows and rng.random() < density:
                set_wall(wall_map, r, c, WALL_DOWN)
    return wall_map


def sample_maps():
    """Hand-built, random and Project A maps used in several tests."""
    return [
        ("open5", open_wall_map(5, 5)),
        ("random4x6", random_maze(4, 6, seed=1)),
        ("random7x7", random_maze(7, 7, seed=2)),
        ("projectA_F1", np.array(get_maps()[0]["wall_map"])),
    ]


# ---------------------------------------------------------------------------
# t3.0: 8 move actions and DECLARE
# ---------------------------------------------------------------------------

def test_action_set():
    assert len(MOVE_ACTIONS) == 8
    assert Action.DECLARE not in MOVE_ACTIONS
    assert len(Action) == 9
    assert len(set(DIR_DELTAS)) == 8
    assert all(max(abs(dr), abs(dc)) == 1 for dr, dc in DIR_DELTAS)
    assert [a.is_diagonal for a in MOVE_ACTIONS] == [False, True] * 4


def test_default_params_match_t1_defaults():
    # t1.3 defaults (docs/decisions/t1_defaults.md). PROJECT_A is the name of
    # the Gaussian-plus-extras form Project A used; its values changed in t1.3.
    assert PROJECT_A.p_drift == pytest.approx(config.MOTION_P_DRIFT) == pytest.approx(1e-4)
    assert PROJECT_A.p_steps == pytest.approx((0.05, 0.9499, 0.0001), abs=1e-6)
    assert HARD.p_steps == pytest.approx((0.1, 0.898, 0.002)) and HARD.p_drift == pytest.approx(0.002)
    assert PROJECT_A.wall_mode == "collide"
    assert MotionParams.from_config(noisy=False) == DET


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_steps": (0.5, 0.5)},
        {"p_steps": (0.5, 0.6, 0.1)},
        {"p_steps": (-0.1, 1.0, 0.1)},
        {"p_drift": 1.5},
        {"wall_mode": "bounce"},
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        MotionParams(**kwargs)


# ---------------------------------------------------------------------------
# t3.1: probabilities sum to 1 for every (state, action)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", ALL_PARAMS, ids=PARAM_IDS)
@pytest.mark.parametrize("name,wall_map", sample_maps(), ids=[m[0] for m in sample_maps()])
def test_transition_rows_sum_to_one(name, wall_map, params):
    T, C = build_transition_model(wall_map, params)
    rows, cols = wall_map.shape[:2]
    assert T.shape == (8, rows * cols, rows * cols)
    assert C.shape == (8, rows * cols)
    assert (T >= 0).all() and (C >= 0).all()
    np.testing.assert_allclose(T.sum(axis=-1) + C, 1.0, atol=1e-12)
    if params.wall_mode == "stop":
        assert (C == 0).all()


def test_deterministic_open_interior():
    wall_map = open_wall_map(5, 5)
    T, C = build_transition_model(wall_map, DET)
    s = cell_index(2, 2, 5)
    for a in MOVE_ACTIONS:
        dr, dc = DIR_DELTAS[a]
        expected = cell_index(2 + dr, 2 + dc, 5)
        assert T[a, s, expected] == 1.0
        assert T[a, s].sum() == 1.0
        assert C[a, s] == 0.0


def test_drift_and_steps_in_open_space():
    wall_map = open_wall_map(7, 7)
    params = MotionParams((0.1, 0.7, 0.2), 0.2)
    dist, p_col = move_distribution(wall_map, 3, 3, Action.E, params)
    assert p_col == 0.0
    assert dist[(3, 3)] == pytest.approx(0.1)            # k = 0, any direction
    assert dist[(3, 4)] == pytest.approx(0.8 * 0.7)      # E, k = 1
    assert dist[(3, 5)] == pytest.approx(0.8 * 0.2)      # E, k = 2
    assert dist[(2, 4)] == pytest.approx(0.1 * 0.7)      # drift NE, k = 1
    assert dist[(4, 4)] == pytest.approx(0.1 * 0.7)      # drift SE, k = 1
    assert dist[(1, 5)] == pytest.approx(0.1 * 0.2)      # drift NE, k = 2
    assert dist[(5, 5)] == pytest.approx(0.1 * 0.2)      # drift SE, k = 2


def test_diagonal_drifts_to_cardinals():
    wall_map = open_wall_map(5, 5)
    params = MotionParams((0.0, 1.0, 0.0), 0.5)
    dist, _ = move_distribution(wall_map, 2, 2, Action.NE, params)
    assert dist == pytest.approx({(1, 3): 0.5, (1, 2): 0.25, (2, 3): 0.25})


# ---------------------------------------------------------------------------
# Wall geometry (D2 diagonal rule, D3 collision semantics)
# ---------------------------------------------------------------------------

def _edge_segment(r, c, wall):
    """Edge of cell (r, c) as a pair of grid points (x, y), x right, y down."""
    return {
        WALL_UP: ((c, r), (c + 1, r)),
        WALL_DOWN: ((c, r + 1), (c + 1, r + 1)),
        WALL_LEFT: ((c, r), (c, r + 1)),
        WALL_RIGHT: ((c + 1, r), (c + 1, r + 1)),
    }[wall]


def _interior_edges(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((r, c, WALL_RIGHT))
            if r + 1 < rows:
                edges.append((r, c, WALL_DOWN))
    return edges


@pytest.mark.parametrize("direction", [a for a in MOVE_ACTIONS if a.is_diagonal], ids=lambda a: a.name)
def test_diagonal_blocked_exactly_by_edges_touching_the_corner(direction):
    """A diagonal is a straight line through the shared corner: any wall that
    touches that corner blocks it, and no other wall does."""
    dr, dc = DIR_DELTAS[direction]
    corner = (1 + (dc > 0), 1 + (dr > 0))   # grid point (x, y) shared by the 4 cells
    for r, c, wall in _interior_edges(3, 3):
        wall_map = open_wall_map(3, 3)
        set_wall(wall_map, r, c, wall)
        touches = corner in _edge_segment(r, c, wall)
        assert can_step(wall_map, 1, 1, direction) == (not touches), (r, c, wall)


def test_diagonal_blocked_when_only_one_l_path_is_open():
    wall_map = open_wall_map(3, 3)
    set_wall(wall_map, 0, 1, WALL_RIGHT)   # blocks N-then-E, E-then-N stays open
    assert not can_step(wall_map, 1, 1, Action.NE)
    assert can_step(wall_map, 1, 1, Action.NW)


def test_border_is_a_wall():
    wall_map = open_wall_map(3, 3)
    for a in MOVE_ACTIONS:
        dr, dc = DIR_DELTAS[a]
        expected = 0 <= dr < 3 and 0 <= dc < 3   # from (0, 0)
        assert can_step(wall_map, 0, 0, a) == expected


def test_two_step_move_checks_every_edge():
    wall_map = open_wall_map(1, 4)
    set_wall(wall_map, 0, 1, WALL_RIGHT)   # wall between col 1 and col 2
    hit = resolve_move(wall_map, 0, 0, Action.E, 2, "collide")
    assert hit.collided and (hit.r, hit.c) == (0, 1) and hit.cells_moved == 1
    stop = resolve_move(wall_map, 0, 0, Action.E, 2, "stop")
    assert not stop.collided and (stop.r, stop.c) == (0, 1) and stop.cells_moved == 1
    ok = resolve_move(wall_map, 0, 0, Action.E, 1, "collide")
    assert not ok.collided and (ok.r, ok.c) == (0, 1)


def test_zero_steps_never_collide():
    wall_map = open_wall_map(2, 2)
    params = MotionParams((1.0, 0.0, 0.0), 0.3)
    for a in MOVE_ACTIONS:
        dist, p_col = move_distribution(wall_map, 0, 0, a, params)
        assert p_col == 0.0 and dist == {(0, 0): pytest.approx(1.0)}


def test_collision_probability_next_to_wall():
    wall_map = open_wall_map(5, 5)
    params = MotionParams((0.1, 0.8, 0.1), 0.2)
    # From the top row, N and both drifts (NW, NE) leave the map for k >= 1.
    _, p_col = move_distribution(wall_map, 0, 2, Action.N, params)
    assert p_col == pytest.approx(0.9)
    # One row down, only k = 2 hits the border.
    _, p_col = move_distribution(wall_map, 1, 2, Action.N, params)
    assert p_col == pytest.approx(0.1)


def _rotate_cw(wall_map):
    """Rotate a wall_map 90° clockwise. Cell (r, c) -> (c, rows-1-r)."""
    rot = np.rot90(wall_map, k=-1, axes=(0, 1))
    out = np.empty_like(rot)
    out[..., WALL_RIGHT] = rot[..., WALL_UP]
    out[..., WALL_DOWN] = rot[..., WALL_RIGHT]
    out[..., WALL_LEFT] = rot[..., WALL_DOWN]
    out[..., WALL_UP] = rot[..., WALL_LEFT]
    return out


def test_rotation_symmetry():
    wall_map = random_maze(4, 6, seed=7)
    rows, cols = wall_map.shape[:2]
    rotated = _rotate_cw(wall_map)
    assert rotated.shape[:2] == (cols, rows)
    T, C = build_transition_model(wall_map, HARD)
    T_rot, C_rot = build_transition_model(rotated, HARD)

    def f(s):
        r, c = index_to_cell(s, cols)
        return cell_index(c, rows - 1 - r, rows)

    perm = np.array([f(s) for s in range(rows * cols)])
    for a in MOVE_ACTIONS:
        a_rot = (int(a) + 2) % 8   # N -> E, NE -> SE, ...
        np.testing.assert_allclose(T_rot[a_rot][np.ix_(perm, perm)], T[a], atol=1e-12)
        np.testing.assert_allclose(C_rot[a_rot][perm], C[a], atol=1e-12)


# ---------------------------------------------------------------------------
# t3.2: empirical distribution within ±1% of the model over 10,000 samples
# ---------------------------------------------------------------------------

N_SAMPLES = 10_000
COLLISION = "collision"


def _model_distribution(wall_map, r, c, a, params):
    dist, p_col = move_distribution(wall_map, r, c, a, params)
    if p_col > 0:
        dist[COLLISION] = p_col
    return dist


def _chi2_critical(df, z=3.09):
    """Wilson-Hilferty approximation of the chi-square quantile (z=3.09 -> 99.9%)."""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def _check_empirical(model, counts, n):
    keys = set(model) | set(counts)
    for key in keys:
        p = model.get(key, 0.0)
        freq = counts.get(key, 0) / n
        assert p > 0 or counts.get(key, 0) == 0, f"sampled impossible outcome {key}"
        assert abs(freq - p) <= 0.01, f"{key}: empirical {freq:.4f} vs model {p:.4f}"
    support = [k for k in model if model[k] > 0]
    if len(support) > 1:
        chi2 = sum((counts.get(k, 0) - n * model[k]) ** 2 / (n * model[k]) for k in support)
        assert chi2 < _chi2_critical(len(support) - 1), f"chi2 {chi2:.1f}"


def _cases():
    """(state, action) pairs in a Project A map: open space, next to walls, corners."""
    wall_map = np.array(get_maps()[0]["wall_map"])
    T, C = build_transition_model(wall_map, HARD)
    cols = wall_map.shape[1]
    # Pick pairs with the most varied outcomes, including collision risk.
    picks = []
    for a in (Action.N, Action.NE, Action.E, Action.SW):
        support = (T[a] > 0).sum(axis=1) + (C[a] > 0)
        risky = np.where((C[a] > 0.05) & (support >= 4))[0]
        safe = np.where((C[a] == 0) & (support >= 7))[0]
        for s in (risky[:1].tolist() + safe[:1].tolist()):
            picks.append((wall_map, *index_to_cell(s, cols), a))
    return picks


@pytest.mark.parametrize("params", [PROJECT_A, HARD], ids=["projectA", "hard"])
def test_sample_move_matches_model(params):
    cases = _cases()
    assert len(cases) >= 6
    for i, (wall_map, r, c, a) in enumerate(cases):
        rng = np.random.default_rng(1000 + i)
        counts = {}
        for _ in range(N_SAMPLES):
            m = sample_move(wall_map, r, c, a, params, rng)
            key = COLLISION if m.collided else (m.r, m.c)
            counts[key] = counts.get(key, 0) + 1
        _check_empirical(_model_distribution(wall_map, r, c, a, params), counts, N_SAMPLES)


def test_grid_env_matches_model():
    """Same check through the environment: 10,000 one-step episodes."""
    wall_map, r, c, a = _cases()[0]
    env = GridEnv(wall_map, goal_cells=[(0, 0)], params=HARD)
    counts = {}
    for seed in range(N_SAMPLES):
        env.reset(seed=seed, start=(r, c))
        res = env.step(a)
        key = COLLISION if res.outcome is Outcome.COLLISION else res.info["cell"]
        counts[key] = counts.get(key, 0) + 1
    _check_empirical(_model_distribution(wall_map, r, c, a, HARD), counts, N_SAMPLES)


# ---------------------------------------------------------------------------
# t3.3: a collision ends the run and is logged as its own failure type
# ---------------------------------------------------------------------------

def test_collision_ends_run_and_is_logged():
    env = GridEnv(open_wall_map(3, 3), goal_cells=[(1, 1)], params=DET)
    env.reset(seed=0, start=(0, 0))
    res = env.step(Action.N)
    assert res.outcome is Outcome.COLLISION
    assert res.outcome.is_terminal and res.outcome.is_failure
    log = res.info["log"]
    assert log.outcome is Outcome.COLLISION
    assert log.collision is not None and log.collision.direction is Action.N
    assert log.final_cell == (0, 0)
    assert log.as_row()["outcome"] == "collision"
    with pytest.raises(RuntimeError):
        env.step(Action.S)


def test_outcomes_are_counted_separately():
    wall_map = open_wall_map(3, 3)
    env = GridEnv(wall_map, goal_cells=[(1, 1)], params=DET, max_steps=3)
    logs = []

    env.reset(seed=0, start=(0, 0)); env.step(Action.N); logs.append(env.log)       # collision
    env.reset(seed=1, start=(0, 0)); env.step(Action.SE); env.step(Action.DECLARE)
    logs.append(env.log)                                                          # success
    env.reset(seed=2, start=(0, 0)); env.step(Action.DECLARE); logs.append(env.log)  # wrong
    env.reset(seed=3, start=(0, 0))
    for a in (Action.E, Action.W, Action.E):                                      # timeout
        res = env.step(a)
    assert res.outcome is Outcome.TIMEOUT
    logs.append(env.log)

    assert outcome_counts(logs) == {
        "success": 1, "wrong_declaration": 1, "collision": 1, "timeout": 1,
    }
    assert [log.failed for log in logs] == [True, False, True, True]


def test_stop_mode_never_collides():
    env = GridEnv(open_wall_map(3, 3), goal_cells=[(1, 1)],
                  params=MotionParams.deterministic("stop"))
    env.reset(seed=0, start=(0, 0))
    res = env.step(Action.N)
    assert res.outcome is Outcome.RUNNING and res.info["cell"] == (0, 0)


def test_random_start_excludes_goal_and_is_seeded():
    env = GridEnv(open_wall_map(3, 3), goal_cells=[(1, 1)], params=HARD)
    starts = {env.reset(seed=s).info["cell"] for s in range(200)}
    assert (1, 1) not in starts and len(starts) == 8

    def run(seed):
        env.reset(seed=seed)
        while not env.done:
            env.step(Action.E if env.log.steps % 2 == 0 else Action.W)
        return env.log.trajectory, env.log.outcome

    assert run(5) == run(5)


def test_robot_apply_move_action():
    wall_map = open_wall_map(4, 4)
    robot = Robot(1, 1)
    rng = np.random.default_rng(0)
    res = robot.apply_move_action(Action.SE, wall_map, rng, params=DET)
    assert (robot.r, robot.c) == (2, 2) == (res.r, res.c) and not res.collided
    res = robot.apply_move_action(Action.SE, wall_map, rng, params=MotionParams((0, 0, 1), 0))
    assert res.collided and (robot.r, robot.c) == (3, 3)


# ---------------------------------------------------------------------------
# GUI manual control: W moves forward along any of the 8 headings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_index", range(8))
def test_heading_to_action_matches_drawn_heading(angle_index):
    """The move must go where the GUI draws the heading (cos/sin, y down)."""
    robot = Robot(0, 0)
    robot.angle_index = angle_index
    rad = math.radians(robot.angle_deg)
    expected = (int(round(math.sin(rad))), int(round(math.cos(rad))))   # (dr, dc)
    assert DIR_DELTAS[heading_to_action(angle_index)] == expected


def test_heading_to_action_names():
    assert [heading_to_action(i).name for i in range(8)] == [
        "E", "SE", "S", "SW", "W", "NW", "N", "NE",
    ]


@pytest.mark.parametrize("angle_index", range(8))
def test_predict_forward_8dir_deterministic(angle_index):
    wall_map = open_wall_map(5, 5)
    belief = np.zeros((5, 5, 8))
    belief[2, 2, angle_index] = 1.0
    out = predict_forward_8dir(belief, wall_map, DET)
    dr, dc = DIR_DELTAS[heading_to_action(angle_index)]
    assert out[2 + dr, 2 + dc, angle_index] == pytest.approx(1.0)
    assert out.sum() == pytest.approx(1.0)


def test_predict_forward_8dir_matches_model_and_conditions_on_survival():
    wall_map = random_maze(5, 6, seed=3)
    rng = np.random.default_rng(4)
    belief = rng.random((5, 6, 8))
    belief /= belief.sum()
    out = predict_forward_8dir(belief, wall_map, HARD)
    T, C = build_transition_model(wall_map, HARD)
    expected = np.stack(
        [(belief[:, :, th].reshape(-1) @ T[heading_to_action(th)]).reshape(5, 6) for th in range(8)],
        axis=2,
    )
    survive = expected.sum()
    assert survive < 1.0                       # some mass collides in this maze
    np.testing.assert_allclose(out, expected / survive, atol=1e-12)


def test_robot_forward_on_diagonal_heading():
    wall_map = open_wall_map(4, 4)
    robot = Robot(2, 1)
    robot.angle_index = 7                      # NE
    res = robot.apply_move_action(heading_to_action(robot.angle_index), wall_map,
                                  np.random.default_rng(0), params=DET)
    assert (robot.r, robot.c) == (1, 2) and not res.collided and robot.angle_index == 7
