"""Tests for the Project B sensing model (roadmap task t4).

Run from this directory:
    python -m pytest test_sensing.py -q
"""

import itertools

import numpy as np
import pytest

from config import WALL_RIGHT
from grid_env import GridEnv
from map_generator import get_maps
from motion_model import Action, MotionParams, Outcome, cell_index, index_to_cell, open_wall_map, set_wall
from sensing import (
    NUM_RAYS,
    SensingParams,
    build_observation_model,
    decode_observation,
    describe_observation,
    encode_observation,
    noiseless_signatures,
    ray_distributions,
    ray_walls,
    sample_observation,
    signature_ambiguity,
)
from test_motion_model import _check_empirical, _rotate_cw, random_maze, sample_maps

NOISY = SensingParams(R=2, p_miss=0.05)
NOISELESS = SensingParams(R=2, p_miss=0.0)
LONG_NOISY = SensingParams(R=3, p_miss=0.3)
ALL_SENSING = [NOISY, NOISELESS, LONG_NOISY, SensingParams(R=0, p_miss=0.1)]
SENSING_IDS = ["R2_miss05", "R2_noiseless", "R3_miss30", "R0_miss10"]


def corridor(cols, walls_right_of=()):
    """1 x cols map with extra walls on the right edge of the given columns."""
    wall_map = open_wall_map(1, cols)
    for c in walls_right_of:
        set_wall(wall_map, 0, c, WALL_RIGHT)
    return wall_map


# ---------------------------------------------------------------------------
# Parameters and encoding
# ---------------------------------------------------------------------------

def test_default_observation_space_size():
    assert NOISY.base == 4 and NOISY.none == 3
    assert NOISY.num_observations == 256   # D5: (R + 2) ** 4


@pytest.mark.parametrize("kwargs", [{"R": -1}, {"R": 1.5}, {"p_miss": -0.1}, {"p_miss": 1.0}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        SensingParams(**kwargs)


def test_from_config_noise_switch():
    assert SensingParams.from_config(noisy=False).p_miss == 0.0
    assert SensingParams.from_config(noisy=True).p_miss > 0.0


def test_encode_decode_round_trip():
    for params in (NOISY, SensingParams(R=1)):
        seen = set()
        for digits in itertools.product(range(params.base), repeat=NUM_RAYS):
            o = encode_observation(digits, params)
            assert decode_observation(o, params) == digits
            seen.add(o)
        assert seen == set(range(params.num_observations))
    assert encode_observation((1, 0, 0, 0), NOISY) == 1          # ray N is the lowest digit
    assert encode_observation((0, 0, 0, 1), NOISY) == 4 ** 3
    assert describe_observation(encode_observation((0, 3, 2, 1), NOISY), NOISY) == "N=0 E=- S=2 W=1"
    with pytest.raises(ValueError):
        encode_observation((4, 0, 0, 0), NOISY)


# ---------------------------------------------------------------------------
# Ray geometry
# ---------------------------------------------------------------------------

def test_ray_walls_open_3x3():
    table = ray_walls(open_wall_map(3, 3), R=2)
    centre = cell_index(1, 1, 3)
    # Every ray from the centre crosses one open edge, then meets the border.
    np.testing.assert_array_equal(table[centre], [[False, True, False]] * 4)
    corner = cell_index(0, 0, 3)
    # N and W: border on the own edge. E and S: border two cells away.
    np.testing.assert_array_equal(
        table[corner], [[True, False, False], [False, False, True], [False, False, True], [True, False, False]]
    )


def test_ray_passes_missed_walls_and_stops_at_border():
    # Walls right of columns 0 and 1; the border is right of column 3.
    table = ray_walls(corridor(4, walls_right_of=(0, 1)), R=5)
    east = 1
    np.testing.assert_array_equal(table[0, east], [True, True, False, True, False, False])


def test_ray_distribution_hand_computed():
    """From column 0: walls at distances 0, 1 and 3 on the east ray."""
    params = SensingParams(R=3, p_miss=0.1)
    probs = ray_distributions(ray_walls(corridor(4, walls_right_of=(0, 1)), 3), params)
    np.testing.assert_allclose(probs[0, 1], [0.9, 0.09, 0.0, 0.009, 0.001])
    # North ray: the border is on the robot's own edge.
    np.testing.assert_allclose(probs[0, 0], [0.9, 0.0, 0.0, 0.0, 0.1])


def test_wall_out_of_range_reads_none():
    params = SensingParams(R=1, p_miss=0.0)
    Z = build_observation_model(open_wall_map(5, 5), params)
    centre = cell_index(2, 2, 5)
    o = int(Z[centre].argmax())
    assert decode_observation(o, params) == (params.none,) * 4


# ---------------------------------------------------------------------------
# Z: shape, stochasticity, noiseless case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", ALL_SENSING, ids=SENSING_IDS)
@pytest.mark.parametrize("name,wall_map", sample_maps(), ids=[m[0] for m in sample_maps()])
def test_observation_rows_sum_to_one(name, wall_map, params):
    Z = build_observation_model(wall_map, params)
    rows, cols = wall_map.shape[:2]
    assert Z.shape == (rows * cols, params.num_observations)
    assert (Z >= 0).all()
    np.testing.assert_allclose(Z.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("name,wall_map", sample_maps(), ids=[m[0] for m in sample_maps()])
def test_noiseless_is_one_hot_and_matches_signature(name, wall_map):
    Z = build_observation_model(wall_map, NOISELESS)
    assert set(np.unique(Z)) <= {0.0, 1.0}
    np.testing.assert_array_equal((Z == 1.0).sum(axis=1), 1)
    np.testing.assert_array_equal(Z.argmax(axis=1), noiseless_signatures(wall_map, NOISELESS.R))


def test_most_likely_observation_is_the_noiseless_one():
    wall_map = random_maze(6, 6, seed=3)
    Z = build_observation_model(wall_map, NOISY)
    np.testing.assert_array_equal(Z.argmax(axis=1), noiseless_signatures(wall_map, NOISY.R))


def test_rotation_symmetry():
    """Rotating the maze clockwise permutes cells and rotates the rays N->E->S->W."""
    wall_map = random_maze(4, 6, seed=7)
    rows, cols = wall_map.shape[:2]
    Z = build_observation_model(wall_map, LONG_NOISY)
    Z_rot = build_observation_model(_rotate_cw(wall_map), LONG_NOISY)

    cell_perm = np.array([cell_index(c, rows - 1 - r, rows) for r, c in (index_to_cell(s, cols) for s in range(rows * cols))])
    obs_perm = np.empty(LONG_NOISY.num_observations, dtype=int)
    for o in range(LONG_NOISY.num_observations):
        n, e, s, w = decode_observation(o, LONG_NOISY)
        obs_perm[o] = encode_observation((w, n, e, s), LONG_NOISY)
    np.testing.assert_allclose(Z_rot[np.ix_(cell_perm, obs_perm)], Z, atol=1e-15)


# ---------------------------------------------------------------------------
# R-limited ambiguity (input to t6.1)
# ---------------------------------------------------------------------------

def test_signature_ambiguity_open_mazes():
    # 5x5 open, R = 2: the (N, S) and (W, E) readings are unique per row/column.
    assert signature_ambiguity(open_wall_map(5, 5), R=2) == {
        "distinct": 25, "ambiguity": 0.0, "largest_group": 1, "ambiguous_cells": 0
    }
    # 9x9 open, R = 2: rows/columns 3..5 all read (NONE, NONE), so the 3x3 centre looks the same.
    amb = signature_ambiguity(open_wall_map(9, 9), R=2)
    assert amb["distinct"] == 7 * 7
    assert amb["largest_group"] == 9
    assert amb["ambiguity"] == pytest.approx(1 - 49 / 81)


def test_signature_ambiguity_grows_as_range_shrinks():
    wall_map = np.array(get_maps()[0]["wall_map"])
    distinct = [signature_ambiguity(wall_map, R)["distinct"] for R in (0, 1, 2, 3)]
    assert distinct == sorted(distinct)


# ---------------------------------------------------------------------------
# Sampling matches Z (the true robot and the filter agree)
# ---------------------------------------------------------------------------

N_SAMPLES = 10_000


def _rich_cells(Z, k=3):
    """Cells with the widest observation distributions."""
    support = (Z > 0).sum(axis=1)
    return np.argsort(-support, kind="stable")[:k]


@pytest.mark.parametrize("params", [NOISY, LONG_NOISY], ids=["R2_miss05", "R3_miss30"])
def test_sample_observation_matches_model(params):
    wall_map = np.array(get_maps()[0]["wall_map"])
    Z = build_observation_model(wall_map, params)
    table = ray_walls(wall_map, params.R)
    for i, s in enumerate(_rich_cells(Z)):
        rng = np.random.default_rng(2000 + i)
        counts = {}
        for _ in range(N_SAMPLES):
            o = sample_observation(table, int(s), params, rng)
            counts[o] = counts.get(o, 0) + 1
        model = {o: float(Z[s, o]) for o in np.flatnonzero(Z[s])}
        _check_empirical(model, counts, N_SAMPLES)


def test_noiseless_sampling_is_deterministic():
    wall_map = random_maze(5, 5, seed=4)
    table = ray_walls(wall_map, 2)
    sig = noiseless_signatures(wall_map, 2)
    rng = np.random.default_rng(0)
    for s in range(25):
        assert sample_observation(table, s, NOISELESS, rng) == sig[s]


# ---------------------------------------------------------------------------
# GridEnv integration (observation timing, roadmap D7)
# ---------------------------------------------------------------------------

def test_env_without_sensing_has_no_observations():
    env = GridEnv(open_wall_map(3, 3), goal_cells=[(1, 1)], params=MotionParams.deterministic())
    assert env.reset(seed=0).obs is None
    assert env.step(Action.N).obs is None


def test_env_observation_timing():
    env = GridEnv(open_wall_map(3, 3), goal_cells=[(1, 1)], params=MotionParams.deterministic(),
                  sensing=NOISELESS)
    sig = noiseless_signatures(env.wall_map, 2)
    res = env.reset(seed=0, start=(2, 1))
    assert res.obs == sig[cell_index(2, 1, 3)]                   # observation at reset
    res = env.step(Action.N)
    assert res.outcome is Outcome.RUNNING and res.obs == sig[cell_index(1, 1, 3)]
    res = env.step(Action.DECLARE)
    assert res.outcome is Outcome.SUCCESS and res.obs is None    # DECLARE: no observation
    assert env.log.observations == [sig[cell_index(2, 1, 3)], sig[cell_index(1, 1, 3)]]


def test_env_collision_gives_no_observation():
    env = GridEnv(open_wall_map(3, 3), goal_cells=[(1, 1)], params=MotionParams.deterministic(),
                  sensing=NOISY)
    env.reset(seed=0, start=(0, 0))
    res = env.step(Action.N)
    assert res.outcome is Outcome.COLLISION and res.obs is None


def test_env_observations_are_seeded():
    def run(seed):
        env = GridEnv(random_maze(5, 5, seed=1), goal_cells=[(2, 2)], params=MotionParams.hard("stop"),
                      sensing=LONG_NOISY)
        obs = [env.reset(seed=seed).obs]
        for a in (Action.N, Action.E, Action.S, Action.W) * 3:
            obs.append(env.step(a).obs)
        return obs, env.log.trajectory

    assert run(5) == run(5)
    assert run(5) != run(6)
