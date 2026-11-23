"""Tests for the ported Project A heuristic (roadmap task t7: t7.0, t7.1, t7.2).

Run from this directory:
    python -m pytest test_projecta.py -q
"""

import json

import numpy as np
import pytest

import maze_bank
from grid_env import GridEnv
from motion_model import Action, MotionParams, Outcome, cell_index
from policies import EnvInfo, ProjectAPolicy, make_policy
from projecta import (
    Model,
    ProjectAAgent,
    ProjectAParams,
    belief_entropy,
    dominant_modes,
    expected_posterior_entropy,
    is_localized,
    score_moves,
)
from runner import LoadedMaze, run_batch, summarize
from sensing import SensingParams, build_observation_model, noiseless_signatures
from sim_config import ConfigError, load_config

CFG = load_config()
PARAMS = ProjectAParams.from_mapping(CFG.projecta)
SMALL = maze_bank.list_mazes("small")


def _model(name="small/simple", motion=None, sensing=None):
    spec = maze_bank.load_maze(name)
    return Model.build(
        spec.wall_map,
        spec.goal_cells,
        motion or CFG.to_motion_params(),
        sensing or CFG.to_sensing_params(),
    )


# ---------------------------------------------------------------------------
# Information measures (Project A's active_localization._demo() checks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 5, 25])
def test_entropy_of_k_equal_cells_is_ln_k(k):
    b = np.zeros(30)
    b[:k] = 1.0 / k
    assert belief_entropy(b) == pytest.approx(np.log(k))


def test_expected_posterior_entropy_matches_brute_force():
    rng = np.random.default_rng(0)
    Z = build_observation_model(maze_bank.load_maze("small/m5_a").wall_map, SensingParams(2, 0.1))
    b = rng.random(25)
    b[rng.random(25) < 0.4] = 0.0
    b /= b.sum()
    brute = 0.0
    for o in range(Z.shape[1]):
        joint = Z[:, o] * b
        p_o = joint.sum()
        if p_o > 0:
            brute += p_o * belief_entropy(joint)
    assert expected_posterior_entropy(b, Z) == pytest.approx(brute)


def test_positive_gain_from_uniform_belief():
    m = _model()
    b = np.full(m.rows * m.cols, 1.0 / (m.rows * m.cols))
    gains = [s.gain for s in score_moves(b, m.T, m.C, m.Z_used, m.moves)]
    assert max(gains) > 0.0


def test_two_mode_belief_gets_resolved():
    """Two distant cells that look the same: one move should tell them apart."""
    det = MotionParams.deterministic()
    clean = SensingParams(2, 0.0)
    for name in SMALL:
        m = _model(name, det, clean)
        sig = noiseless_signatures(m.wall_map, 2)
        pairs = [
            (i, j) for i in range(sig.size) for j in range(i + 1, sig.size)
            if sig[i] == sig[j] and max(abs(i // m.cols - j // m.cols), abs(i % m.cols - j % m.cols)) >= 2
        ]
        if not pairs:
            continue
        i, j = pairs[0]
        b = np.zeros(sig.size)
        b[[i, j]] = 0.5
        assert not is_localized(b, m.rows, m.cols, PARAMS)
        best = max((s for s in score_moves(b, m.T, m.C, m.Z_used, m.moves) if s.risk == 0.0),
                   key=lambda s: s.gain, default=None)
        if best is not None and best.gain > 0:
            assert best.gain == pytest.approx(np.log(2))
            return
    pytest.skip("no small maze has a resolvable distant look-alike pair")


# ---------------------------------------------------------------------------
# Switching rule (t1.2) and parameters (t7.0)
# ---------------------------------------------------------------------------

def test_dominant_modes_ignore_adjacent_rival():
    b = np.zeros(25)
    b[cell_index(2, 2, 5)] = 0.5
    b[cell_index(2, 3, 5)] = 0.45   # Chebyshev distance 1: not a rival
    b[cell_index(0, 0, 5)] = 0.05
    peak, p_max, p_far = dominant_modes(b, 5, 5, sep_min=2)
    assert peak == cell_index(2, 2, 5) and p_max == 0.5 and p_far == 0.05
    assert is_localized(b, 5, 5, PARAMS)


def test_distant_rival_blocks_localization():
    b = np.zeros(25)
    b[cell_index(0, 0, 5)] = 0.5
    b[cell_index(4, 4, 5)] = 0.4
    b[cell_index(2, 2, 5)] = 0.1
    assert not is_localized(b, 5, 5, PARAMS)


def test_thresholds_are_config_parameters():
    """t7.0: switching criterion and declaration threshold come from the config."""
    names = {"tau_loc", "rho_amb", "tau_lost", "sep_min", "tau_declare", "eps_col", "move_cost"}
    assert names == set(PARAMS.as_dict())
    cfg = CFG.replace(["policy.name=projecta", "projecta.tau_declare=0.99", "projecta.tau_loc=0.4"])
    policy = make_policy(cfg.policy.name, cfg.policy.selected_params, cfg=cfg)
    assert policy.thresholds.tau_declare == 0.99 and policy.thresholds.tau_loc == 0.4
    maze = LoadedMaze.from_spec(maze_bank.load_maze("small/simple"))
    row = run_batch(cfg.replace(["run.n_runs=1"]), maze)[0]
    assert json.loads(row["policy_params_json"])["tau_declare"] == 0.99
    assert row["projecta.tau_declare"] == 0.99


def test_hysteresis_is_enforced():
    with pytest.raises(ValueError):
        ProjectAParams.from_mapping({**PARAMS.as_dict(), "tau_lost": 0.3})
    with pytest.raises(ConfigError):
        CFG.replace(["projecta.tau_lost=0.3"])


def test_needs_sensing():
    spec = maze_bank.load_maze("small/simple")
    env = GridEnv(spec.wall_map, spec.goal_cells, params=CFG.to_motion_params())
    with pytest.raises(ValueError, match="sensing"):
        ProjectAPolicy().reset(EnvInfo.from_env(env), np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Behaviour on episodes
# ---------------------------------------------------------------------------

def _episodes(name, n, sensing=None):
    spec = maze_bank.load_maze(name)
    env = GridEnv(spec.wall_map, spec.goal_cells, params=CFG.to_motion_params(),
                  max_steps=CFG.episode.max_steps, sensing=sensing or CFG.to_sensing_params())
    policy = ProjectAPolicy(thresholds=PARAMS)
    for seed in range(n):
        res = env.reset(seed)
        policy.reset(EnvInfo.from_env(env), np.random.default_rng(seed))
        while not env.done:
            a = policy.act(res.obs, {})
            if a is Action.DECLARE:
                assert policy.agent.p_goal >= PARAMS.tau_declare
            res = env.step(a)
        yield env.log, policy.agent


def test_never_declares_below_tau_declare_and_respects_gate():
    for log, agent in _episodes("small/m5_c", 100):
        if log.outcome in (Outcome.SUCCESS, Outcome.WRONG_DECLARATION):
            assert agent.p_goal >= PARAMS.tau_declare
        assert agent.n_fallbacks == 0          # the model matches GridEnv exactly


def test_true_cell_keeps_positive_belief():
    for log, agent in _episodes("small/m5_b", 50):
        if log.outcome is Outcome.SUCCESS:
            r, c = log.final_cell
            assert agent.belief[cell_index(r, c, 5)] > 0.0


@pytest.mark.parametrize("name", SMALL)
def test_success_rate_on_5x5(name):
    """t7.1: at least 90% success on a 5x5 maze with the default parameters (500 seeds)."""
    cfg = CFG.replace(["policy.name=projecta", "run.n_runs=500", "run.seed0=0"])
    s = summarize(run_batch(cfg, LoadedMaze.from_spec(maze_bank.load_maze(name))))
    assert s["outcomes"]["success"] / s["n"] >= 0.90, s


def test_runs_on_every_maze_through_the_runner():
    """t7.2: the same config runs on every maze in the bank, only maze.name changes."""
    for name in maze_bank.list_mazes():
        cfg = CFG.replace(["policy.name=projecta", "run.n_runs=3", f"maze.name={name}"])
        rows = run_batch(cfg)
        assert len(rows) == 3
        assert all(r["outcome"] != Outcome.RUNNING.value for r in rows)
        assert all(r["final_P_goal"] is not None and r["n_fallbacks"] == 0 for r in rows)
