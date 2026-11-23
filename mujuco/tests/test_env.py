"""MazeSim (m1, noise off) and the m2 sensor model."""

import numpy as np
import pytest

from sim.env import MazeSim, discretize_ranges
from sim.mjcf import CELL, SENSOR_MAX_RANGE, WALL_OFFSET
from sim.noise import NoiseConfig, corrupt_ranges
from sim.noisy_env import NoisyMazeSim
import maze_bank
from config import WALL_RIGHT
from motion_model import DIR_DELTAS, MOVE_ACTIONS, Action, can_step
from sensing import noiseless_signatures


@pytest.fixture
def smoke_sim():
    wm = maze_bank.open_wall_map(4, 4)
    maze_bank.set_edge(wm, 1, 1, WALL_RIGHT, 0)
    return MazeSim(wm, start=(1, 1), R=3)


def test_smoke_sequence(smoke_sim):
    """Setup guide §5, with rays in sensing.py order (N, E, S, W) and no corner posts."""
    sim = smoke_sim
    assert sim.cell() == (1, 1)
    np.testing.assert_allclose(sim.ranges(), [0.265, 0.085, 0.445, 0.265], atol=1e-3)
    assert sim.observe().tolist() == [1, 0, 2, 1]

    res = sim.step_action(Action.E, 1)            # inner wall
    assert (res.cell, res.collided) == ((1, 1), True)
    assert sim.observe().tolist() == [1, 0, 2, 1]

    res = sim.step_action(Action.S, 2)            # recenter() cleared the contact
    assert (res.cell, res.collided) == ((3, 1), False)
    assert sim.observe().tolist() == [3, 2, 0, 1]

    res = sim.step_action(Action.E, 2)
    assert (res.cell, res.collided) == ((3, 3), False)
    assert sim.observe().tolist() == [3, 0, 0, 3]

    res = sim.step_action(Action.NW, 1)           # empty corner: the diagonal passes
    assert (res.cell, res.collided) == ((2, 2), False)


@pytest.mark.parametrize("name", maze_bank.list_mazes("small"))
def test_every_move_matches_the_D2_rule(name):
    """No posts: from every cell, a noiseless one-cell move in each of the 8 directions
    collides in MuJoCo exactly when motion_model.can_step says it is blocked (D2), and
    otherwise lands on the target cell."""
    wm = maze_bank.load_maze(name).wall_map
    rows, cols = wm.shape[:2]
    sim = MazeSim(wm, start=(0, 0), R=2)
    for r in range(rows):
        for c in range(cols):
            for a in MOVE_ACTIONS:
                sim.teleport((r, c))
                res = sim.step_action(a, 1)
                legal = can_step(wm, r, c, a)
                assert res.collided == (not legal), (name, (r, c), a.name)
                dr, dc = DIR_DELTAS[int(a)]
                assert res.cell == ((r + dr, c + dc) if legal else (r, c)), (name, (r, c), a.name)


@pytest.mark.parametrize("name", maze_bank.list_mazes())
def test_noiseless_observation_matches_sensing_model(name):
    """Every cell of every bank maze: the MuJoCo rays give exactly sensing.py's
    noiseless observation (wall_map adapter, ray order and range all agree)."""
    spec = maze_bank.load_maze(name)
    R = 2
    sim = MazeSim(spec.wall_map, start=(0, 0), R=R)
    expected = noiseless_signatures(spec.wall_map, R)
    rows, cols = spec.wall_map.shape[:2]
    for r in range(rows):
        for c in range(cols):
            sim.teleport((r, c))
            assert sim.observe_index() == expected[r * cols + c], (r, c)


@pytest.mark.parametrize("a", [Action.N, Action.E, Action.S, Action.W])
@pytest.mark.parametrize("k", [1, 2])
def test_noiseless_free_step_lands_exactly_k_cells(a, k):
    wm = maze_bank.open_wall_map(5, 5)
    sim = MazeSim(wm, start=(2, 2), R=2)
    dr, dc = DIR_DELTAS[int(a)]
    res = sim.step_action(a, k)
    assert not res.collided and not res.timed_out
    assert res.cell == (2 + k * dr, 2 + k * dc)


def test_sensor_never_reports_nothing():
    """A ray with no wall within the 1.3 m sensor range reads the maximum range."""
    wm = maze_bank.open_wall_map(1, 12)            # a 12-cell corridor: E from (0,0) is 2.06 m
    sim = MazeSim(wm, start=(0, 0), R=2)
    raw = sim.raw_ranges()[1]                      # MuJoCo's own "no hit": -1, or the cutoff (4.0.1)
    assert raw < 0 or raw == pytest.approx(SENSOR_MAX_RANGE)
    assert sim.ranges()[1] == SENSOR_MAX_RANGE
    assert (sim.ranges() > 0).all()
    assert sim.observe()[1] == 3                   # far: no wall within R = 2


def test_discretize_rounds_and_maps_far_readings():
    R = 2
    r = np.array([WALL_OFFSET + CELL - 1e-9, WALL_OFFSET + 2 * CELL + 0.05, SENSOR_MAX_RANGE, WALL_OFFSET - 0.01])
    assert discretize_ranges(r, R).tolist() == [1, 2, R + 1, 0]
    assert discretize_ranges([WALL_OFFSET + 2.6 * CELL], R).tolist() == [R + 1]


def test_bad_returns_are_wrong_readings_never_empty():
    rng = np.random.default_rng(0)
    cfg = NoiseConfig(bad_return_on=True, p_bad_return=0.3, bad_return_sigma=0.15)
    true = np.full(20000, WALL_OFFSET + CELL)       # a wall one cell away
    noisy = corrupt_ranges(true, cfg, rng)
    assert (noisy >= 0.001).all() and (noisy <= SENSOR_MAX_RANGE).all()
    changed = noisy != true
    assert changed.mean() == pytest.approx(0.3, abs=0.02)
    digits = discretize_ranges(noisy, 2)
    wrong = (digits != 1).mean()
    assert 0.1 < wrong < 0.3                        # most bad returns change the digit, some do not
    assert set(np.unique(digits)) <= {0, 1, 2, 3}


def test_noise_off_changes_nothing():
    true = np.array([0.085, 0.265, 1.3])
    assert (corrupt_ranges(true, NoiseConfig.all_off(), np.random.default_rng(0)) == true).all()


def test_noisy_sim_with_noise_off_matches_m1():
    wm = maze_bank.open_wall_map(5, 5)
    noisy = NoisyMazeSim(wm, start=(2, 2), R=2, noise=NoiseConfig.all_off(), seed=0)
    res = noisy.step_action(Action.E, 1)
    assert (res.cell, res.collided) == ((2, 3), False)
    assert noisy.observe().tolist() == MazeSim(wm, start=(2, 3), R=2).observe().tolist()


def test_noisy_sim_is_reproducible_by_seed():
    wm = maze_bank.open_wall_map(6, 6)
    runs = []
    for _ in range(2):
        sim = NoisyMazeSim(wm, start=(2, 2), R=2, noise=NoiseConfig.all_on(), seed=7)
        runs.append([(sim.step_action(a, 1).cell, tuple(sim.observe())) for a in (Action.E, Action.S, Action.W, Action.N)])
    assert runs[0] == runs[1]
