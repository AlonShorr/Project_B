"""Tests for the exact Bayes filter (roadmap task t4: t4.1, t4.2).

Run from this directory:
    python -m pytest test_belief.py -q
"""

import numpy as np
import pytest

from belief import BayesFilter, correct, observation_probabilities, predict, uniform_belief
from grid_env import GridEnv
from map_generator import get_maps
from motion_model import (
    MOVE_ACTIONS,
    Action,
    MotionParams,
    Outcome,
    build_transition_model,
    cell_index,
    open_wall_map,
)
from sensing import SensingParams, build_observation_model, noiseless_signatures
from test_motion_model import random_maze

NOISY = SensingParams(R=2, p_miss=0.05)
NOISELESS = SensingParams(R=2, p_miss=0.0)
HARD = MotionParams.hard()


def _models(wall_map, motion=HARD, sensing=NOISY):
    T, C = build_transition_model(wall_map, motion)
    return T, C, build_observation_model(wall_map, sensing)


def _random_belief(rng, n, sparse=False):
    b = rng.random(n)
    if sparse:
        b[rng.random(n) < 0.7] = 0.0
        if b.sum() == 0.0:
            b[rng.integers(n)] = 1.0
    return b / b.sum()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_uniform_belief_excludes_goal():
    b = uniform_belief(3, 3, exclude=[(1, 1)])
    assert b[cell_index(1, 1, 3)] == 0.0
    np.testing.assert_allclose(b.sum(), 1.0)
    np.testing.assert_allclose(b[b > 0], 1 / 8)
    with pytest.raises(ValueError):
        uniform_belief(1, 1, exclude=[(0, 0)])


def test_predict_conditions_on_survival():
    wall_map = random_maze(4, 5, seed=2)
    T, C, _ = _models(wall_map)
    rng = np.random.default_rng(0)
    for a in MOVE_ACTIONS:
        b = _random_belief(rng, 20)
        b_bar, ok = predict(b, T, C, a)
        assert ok
        survive = 1.0 - b @ C[a]
        np.testing.assert_allclose(b_bar, (b @ T[a]) / survive)
        np.testing.assert_allclose(b_bar.sum(), 1.0)


def test_predict_rejects_declare():
    T, C, _ = _models(open_wall_map(2, 2))
    with pytest.raises(ValueError):
        predict(uniform_belief(2, 2), T, C, Action.DECLARE)


def test_correct_is_bayes_rule():
    wall_map = random_maze(4, 4, seed=5)
    _, _, Z = _models(wall_map)
    rng = np.random.default_rng(1)
    b = _random_belief(rng, 16)
    p_obs = observation_probabilities(b, Z)
    np.testing.assert_allclose(p_obs.sum(), 1.0)
    for o in np.flatnonzero(p_obs)[:20]:
        post, ok = correct(b, Z, o)
        assert ok
        np.testing.assert_allclose(post, Z[:, o] * b / p_obs[o])


# ---------------------------------------------------------------------------
# t4.1: the belief stays normalized after every update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_normalized_after_random_updates(sparse):
    wall_map = random_maze(5, 5, seed=8)
    T, C, Z = _models(wall_map)
    rng = np.random.default_rng(3)
    f = BayesFilter(T, C, Z)
    f.reset(_random_belief(rng, 25, sparse))
    for _ in range(300):
        # Any observation, possible or not: the result must still be a distribution.
        b = f.update(Action(int(rng.integers(8))), int(rng.integers(Z.shape[1])))
        assert (b >= 0).all()
        np.testing.assert_allclose(b.sum(), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "name,wall_map",
    [("random6x6", random_maze(6, 6, seed=11)), ("projectA_F1", np.array(get_maps()[0]["wall_map"]))],
    ids=["random6x6", "projectA_F1"],
)
def test_filter_tracks_grid_env(name, wall_map):
    """t4.1 along simulated runs: the belief stays normalized, never rules out
    the true cell, and never needs the fallback (the model matches the world)."""
    motion = MotionParams.hard("collide")
    T, C, Z = _models(wall_map, motion, NOISY)
    rows, cols = wall_map.shape[:2]
    goal = (rows // 2, cols // 2)
    env = GridEnv(wall_map, goal_cells=[goal], params=motion, max_steps=40, sensing=NOISY)
    f = BayesFilter(T, C, Z)
    steps = 0
    for seed in range(40):
        rng = np.random.default_rng(10_000 + seed)
        res = env.reset(seed=seed)
        b = f.reset(uniform_belief(rows, cols, exclude=[goal]), res.obs)
        while True:
            assert b[cell_index(*env.cell, cols)] > 0.0
            np.testing.assert_allclose(b.sum(), 1.0, atol=1e-12)
            # Random action among the safest from the true cell, so runs last.
            # The update is exact whatever rule picks the action.
            risk = C[:, cell_index(*env.cell, cols)]
            a = Action(int(rng.choice(np.flatnonzero(risk == risk.min()))))
            res = env.step(a)
            if res.outcome is not Outcome.RUNNING:
                break
            b = f.update(a, res.obs)
            steps += 1
        assert f.n_fallbacks == 0
    assert steps > 100


# ---------------------------------------------------------------------------
# t4.2: with no misses, one observation separates cells with different signatures
# ---------------------------------------------------------------------------

def test_one_observation_separates_different_signatures():
    wall_map = random_maze(5, 5, seed=4)
    Z = build_observation_model(wall_map, NOISELESS)
    sig = noiseless_signatures(wall_map, NOISELESS.R)
    pairs = 0
    for s1 in range(25):
        for s2 in range(25):
            if sig[s1] == sig[s2]:
                continue
            b = np.zeros(25)
            b[[s1, s2]] = 0.5
            post, ok = correct(b, Z, sig[s1])
            assert ok
            assert post[s2] == 0.0 and post[s1] == 1.0
            pairs += 1
    assert pairs > 0


def test_same_signature_is_not_separated():
    """Control: cells that look the same keep their prior ratio."""
    wall_map = open_wall_map(9, 9)          # the 3x3 centre looks the same with R = 2
    Z = build_observation_model(wall_map, NOISELESS)
    a, b_ = cell_index(3, 3, 9), cell_index(5, 5, 9)
    b = np.zeros(81)
    b[a], b[b_] = 0.25, 0.75
    post, _ = correct(b, Z, int(Z[a].argmax()))
    np.testing.assert_allclose(post[[a, b_]], [0.25, 0.75])


def test_noiseless_observation_leaves_only_matching_cells():
    wall_map = np.array(get_maps()[0]["wall_map"])
    Z = build_observation_model(wall_map, NOISELESS)
    sig = noiseless_signatures(wall_map, NOISELESS.R)
    n = sig.size
    for s in range(0, n, 17):
        post, _ = correct(np.full(n, 1 / n), Z, sig[s])
        np.testing.assert_array_equal(post > 0, sig == sig[s])


# ---------------------------------------------------------------------------
# Impossible observation / survival: uniform-mixing fallback, counted
# ---------------------------------------------------------------------------

def test_impossible_observation_falls_back_and_is_counted():
    wall_map = random_maze(5, 5, seed=4)
    T, C, Z = _models(wall_map, sensing=NOISELESS)
    sig = noiseless_signatures(wall_map, 2)
    s_true, s_other = 0, int(np.flatnonzero(sig != sig[0])[0])
    f = BayesFilter(T, C, Z)
    b0 = np.zeros(25)
    b0[s_true] = 1.0
    b = f.reset(b0, int(sig[s_other]))      # impossible: the belief says s_true
    assert f.n_fallbacks == 1
    np.testing.assert_allclose(b.sum(), 1.0)
    np.testing.assert_array_equal(b > 0, sig == sig[s_other])   # only consistent cells survive


def test_observation_impossible_everywhere_gives_uniform():
    wall_map = open_wall_map(3, 3)
    Z = build_observation_model(wall_map, NOISELESS)
    never = np.flatnonzero(Z.sum(axis=0) == 0)[0]
    post, ok = correct(np.full(9, 1 / 9), Z, never)
    assert not ok
    np.testing.assert_allclose(post, 1 / 9)


def test_impossible_survival_falls_back():
    wall_map = open_wall_map(3, 3)
    T, C, Z = _models(wall_map, motion=MotionParams.deterministic())
    f = BayesFilter(T, C, Z)
    b0 = np.zeros(9)
    b0[cell_index(0, 0, 3)] = 1.0             # moving N from the top row always collides
    f.reset(b0)
    b = f.update(Action.N, None)
    assert f.n_fallbacks == 1
    np.testing.assert_allclose(b.sum(), 1.0)
    assert (b > 0).sum() > 1


def test_filter_checks_shapes_and_b0():
    T, C, Z = _models(open_wall_map(2, 2))
    with pytest.raises(ValueError):
        BayesFilter(T, C, Z[:3])
    f = BayesFilter(T, C, Z)
    with pytest.raises(RuntimeError):
        f.update(Action.N, 0)
    with pytest.raises(ValueError):
        f.reset(np.array([0.5, 0.5, 0.5, 0.5]))
