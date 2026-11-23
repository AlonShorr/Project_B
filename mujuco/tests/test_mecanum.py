"""Mecanum Level A: kinematics, the proposal's fit checks 1, 2, 3 and 5, ToF, decisions 1 and 2."""

import numpy as np
import pytest

from sim.mecanum import MecanumNoise, MecanumSim, RobotConfig, Trapezoid, fk, ik
from sim.mjcf import CELL, cell_center_world
import maze_bank
from motion_model import DIR_DELTAS, MOVE_ACTIONS, Action, can_step
from sensing import noiseless_signatures

RC = RobotConfig()


# ---------------------------------------------------------------- kinematics

def test_ik_fk_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        tw = rng.normal(size=3)
        np.testing.assert_allclose(fk(ik(tw, RC.wheel_r, RC.L), RC.wheel_r, RC.L), tw, atol=1e-12)


def test_wheel_patterns():
    r, L = RC.wheel_r, RC.L
    fwd = ik((1, 0, 0), r, L)
    assert np.allclose(fwd, fwd[0])                        # all wheels forward
    strafe = ik((0, 1, 0), r, L)                           # left strafe: FL, RR back; FR, RL forward
    assert strafe[0] < 0 and strafe[3] < 0 and strafe[1] > 0 and strafe[2] > 0
    diag = ik((1, 1, 0), r, L)                             # diagonal: one opposite pair only
    assert diag[0] == pytest.approx(0) and diag[3] == pytest.approx(0) and diag[1] > 0 and diag[2] > 0


def test_trapezoid_profile():
    p = Trapezoid(0.18, 0.30, 1.0)
    assert p.T == pytest.approx(0.9) and p(p.T)[0] == pytest.approx(0.18) and p(0.5)[1] == pytest.approx(0.3)
    tri = Trapezoid(0.05, 0.30, 1.0)                       # too short to reach v_max
    assert tri.v < 0.30 and tri(tri.T)[0] == pytest.approx(0.05)


def test_robot_model_numbers():
    s = MecanumSim(maze_bank.open_wall_map(3, 3), (1, 1))
    assert s.model.body_subtreemass[1] == pytest.approx(0.300)
    assert RC.counts_per_rev == pytest.approx(617.4)
    assert RC.tof_d0 == pytest.approx(0.050)
    assert s.model.nu == 6                                  # 2 planar servos + 4 motors


# ---------------------------------------------------------------- fit checks

@pytest.mark.parametrize("name", maze_bank.list_mazes())
def test_fit1_spawn_everywhere_without_contact(name):
    wm = maze_bank.load_maze(name).wall_map
    s = MecanumSim(wm, (0, 0))
    rows, cols = wm.shape[:2]
    for r in range(rows):
        for c in range(cols):
            s.teleport((r, c))
            assert not s.in_wall_contact(), (r, c)
            assert s.cell() == (r, c)


@pytest.mark.parametrize("name", maze_bank.list_mazes("small"))
def test_fit2_3_every_move_matches_D2(name):
    """Noise off: every cardinal and diagonal one-cell move from every cell collides
    exactly when can_step (D2, no posts) says it is blocked, and otherwise lands
    within 2 mm of the target centre with no contact."""
    wm = maze_bank.load_maze(name).wall_map
    rows, cols = wm.shape[:2]
    s = MecanumSim(wm, (0, 0))
    for r in range(rows):
        for c in range(cols):
            for a in MOVE_ACTIONS:
                s.teleport((r, c))
                res = s.step_action(a, 1)
                legal = can_step(wm, r, c, a)
                assert res.collided == (not legal) and not res.timed_out, (name, (r, c), a.name)
                if legal:
                    dr, dc = DIR_DELTAS[int(a)]
                    assert res.cell == (r + dr, c + dc)
                    assert np.linalg.norm(s.position() - cell_center_world(r + dr, c + dc)) < 0.002


def test_two_cell_move_and_blocked_wheels_keep_counting():
    wm = maze_bank.open_wall_map(5, 5)
    s = MecanumSim(wm, (4, 2))
    res = s.step_action(Action.N, 2)
    assert (res.cell, res.collided) == ((2, 2), False)
    # Into the border, odometry only (front alignment would see the wall and stop pushing):
    s = MecanumSim(wm, (0, 2), controller={"front_alignment": False})
    before, pos = s.encoder_counts(), s.position()
    res = s.step_action(Action.N, 1)
    assert res.collided and res.cell == (0, 2)
    turned = np.abs(s.encoder_counts() - before).mean() / RC.counts_per_rev * 2 * np.pi * RC.wheel_r
    assert turned > 0.15                                    # encoders count most of a cell...
    assert abs(s.position()[1] - pos[1]) < 0.035            # ...the chassis moved < 35 mm (30 mm clearance)
    # With front alignment the front sensor reports the wall, so the wheels stop early.
    s = MecanumSim(wm, (0, 2))
    before = s.encoder_counts()
    assert s.step_action(Action.N, 1).collided
    turned_fa = np.abs(s.encoder_counts() - before).mean() / RC.counts_per_rev * 2 * np.pi * RC.wheel_r
    assert turned_fa < turned


@pytest.mark.parametrize("name", maze_bank.list_mazes())
def test_fit5_tof_noiseless_observation_matches_sensing(name):
    """15 degree cone (default): every cell of every bank maze gives sensing.py's noiseless
    observation at R = 2, and no reading is below the 50 mm cell-centre minimum."""
    spec = maze_bank.load_maze(name)
    s = MecanumSim(spec.wall_map, (0, 0), R=2)
    expected = noiseless_signatures(spec.wall_map, 2)
    rows, cols = spec.wall_map.shape[:2]
    for r in range(rows):
        for c in range(cols):
            s.teleport((r, c))
            assert s.observe_index() == expected[r * cols + c], (r, c)
            assert s.ranges().min() >= 0.049


def test_wide_cone_sees_corridor_side_walls():
    """Proposal D5: at 27 degrees the cone edge hits a 170 mm corridor's side walls, so a
    wall 3 cells down a corridor reads far too short; at 15 degrees it reads correctly."""
    true_k3 = RC.tof_d0 + 3 * CELL
    wm4 = maze_bank.open_wall_map(1, 4)                    # corridor, E wall 3 cells away
    narrow = MecanumSim(wm4, (0, 0), R=3).ranges()[1]
    wide = MecanumSim(wm4, (0, 0), R=3, tof={"fov_deg": 27.0}).ranges()[1]
    assert abs(narrow - true_k3) < 0.06 and wide < true_k3 - 0.15


def test_center_ray_model_is_exact():
    wm = maze_bank.open_wall_map(1, 4)
    s = MecanumSim(wm, (0, 0), R=3, tof={"model": "center_ray"})
    np.testing.assert_allclose(s.ranges(), [RC.tof_d0, RC.tof_d0 + 3 * CELL, RC.tof_d0, RC.tof_d0], atol=1e-6)


# ---------------------------------------------------------------- noise

def test_readings_are_never_empty_with_noise_on():
    s = MecanumSim(maze_bank.load_maze("small/m5_a").wall_map, (0, 0), R=2,
                   noise=MecanumNoise.all_on(p_bad_return=0.5), seed=1)
    for r in range(5):
        for c in range(5):
            s.teleport((r, c))
            readings, spread = s.tof.read(s.data, s.position())
            noisy = s.tof.corrupt(readings, spread, s.noise, s.rng)
            assert (noisy > 0).all() and (noisy <= 1.3).all()
            assert set(s.observe().tolist()) <= {0, 1, 2, 3}


def test_noise_on_is_reproducible_by_seed():
    wm = maze_bank.open_wall_map(6, 6)
    runs = []
    for _ in range(2):
        s = MecanumSim(wm, (2, 2), noise=MecanumNoise.all_on(), seed=3)
        runs.append([(s.step_action(a, 1).cell, tuple(s.observe()), tuple(np.round(s.position(), 9)))
                     for a in (Action.E, Action.S, Action.W, Action.N)])
    assert runs[0] == runs[1]


def test_strafe_efficiency_makes_sideways_moves_short():
    wm = maze_bank.open_wall_map(5, 5)
    s = MecanumSim(wm, (2, 1), controller={"strafe_compensation": False}, noise=MecanumNoise(strafe_on=True, strafe_eff_sigma=0.0))   # eta_y = 0.90
    s.step_action(Action.E, 1)
    short = cell_center_world(2, 2)[0] - s.position()[0]
    assert short == pytest.approx(0.1 * CELL, abs=0.003)                     # 18 mm short, odometry unaware
    s.teleport((2, 1))
    s.step_action(Action.N, 1)                                               # forward is unaffected
    assert np.linalg.norm(s.position() - cell_center_world(1, 1)) < 0.002


# ---------------------------------------------------------------- yaw (D1) and wall centering

def _ns_corridor(rows=5):
    """A one-column N-S corridor: side walls everywhere."""
    from config import WALL_DOWN
    wm = maze_bank.closed_wall_map(rows, 1)
    for r in range(rows - 1):
        maze_bank.set_edge(wm, r, 0, WALL_DOWN, 1)
    return wm


def test_held_mode_without_noise_matches_locked():
    wm = maze_bank.open_wall_map(5, 5)
    held = MecanumSim(wm, (2, 2), controller={"yaw_mode": "held"})
    assert held.model.nq == 3 + 4                           # x, y, yaw + 4 wheels
    for a in (Action.N, Action.E, Action.SW):
        res = held.step_action(a, 1)
        assert not res.collided
        assert np.linalg.norm(held.position() - cell_center_world(*res.cell)) < 0.002
        assert abs(held.yaw()) < 1e-3


def test_held_mode_gyro_heading_hold_keeps_drift_small():
    wm = maze_bank.open_wall_map(5, 5)
    s = MecanumSim(wm, (2, 2), controller={"yaw_mode": "held"},
                   noise=MecanumNoise(imu_on=True, yaw_drift_on=True), seed=1)
    for i in range(10):
        s.step_action((Action.N, Action.S)[i % 2], 1)
    assert 0 < abs(s.yaw()) < np.radians(2)                 # some drift, held within 2 degrees
    gz, acc = s.imu()
    assert np.isfinite(gz) and acc[2] == pytest.approx(9.81, abs=0.2)   # proper acceleration incl. gravity


def test_held_mode_without_heading_hold_drifts_more():
    wm = maze_bank.open_wall_map(5, 5)
    yaws = []
    for kp in (10.0, 0.0):
        s = MecanumSim(wm, (2, 2), controller={"yaw_mode": "held", "kp_heading": kp},
                       noise=MecanumNoise(yaw_drift_on=True), seed=4)
        for i in range(10):
            s.step_action((Action.N, Action.S)[i % 2], 1)
        yaws.append(abs(s.yaw()))
    assert yaws[1] > 3 * yaws[0]


@pytest.mark.parametrize("centering", [False, True])
def test_wall_centering_removes_a_lateral_offset(centering):
    import mujoco
    s = MecanumSim(_ns_corridor(), (4, 0), controller={"wall_centering": centering})
    s.data.qpos[s.q_xy[0]] += 0.020                        # 20 mm east of the centre line
    mujoco.mj_forward(s.model, s.data)
    res = s.step_action(Action.N, 2)
    lateral = s.position()[0] - cell_center_world(*res.cell)[0]
    if centering:
        assert abs(lateral) < 0.002 and s.n_centering > 0
    else:
        assert lateral == pytest.approx(0.020, abs=0.002)


def test_wall_centering_stops_strafe_shortfall_collisions():
    """With strafe efficiency 0.90 each sideways move ends 18 mm short. In a corridor the
    next forward move then scrapes a side wall unless wall centering corrects it."""
    from config import WALL_DOWN
    from config import WALL_RIGHT
    wm = maze_bank.closed_wall_map(4, 3)                     # bottom row open E-W; every column open N-S;
    for c in range(2):                                       # column 2 is a corridor (walls on both sides)
        maze_bank.set_edge(wm, 3, c, WALL_RIGHT, 1)
    maze_bank.set_edge(wm, 0, 0, WALL_RIGHT, 1)
    for c in range(3):
        for r in range(3):
            maze_bank.set_edge(wm, r, c, WALL_DOWN, 1)
    collided = {}
    for centering in (False, True):
        s = MecanumSim(wm, (3, 0), controller={"wall_centering": centering, "strafe_compensation": False, "front_alignment": False},
                       noise=MecanumNoise(strafe_on=True, strafe_eff_sigma=0.0))
        r1 = s.step_action(Action.E, 1)
        r2 = s.step_action(Action.E, 1)                      # 36 mm short of (3, 2)'s centre
        r3 = s.step_action(Action.N, 2)                      # up the 170 mm corridor
        collided[centering] = r1.collided or r2.collided or r3.collided
    assert collided == {False: True, True: False}


def test_on_step_hook_is_called():
    s = MecanumSim(maze_bank.open_wall_map(3, 3), (1, 1))
    calls = []
    s.on_step = lambda data: calls.append(data.time)
    s.step_action(Action.N, 1)
    assert len(calls) > 500 and calls == sorted(calls)


# ---------------------------------------------------------------- strafe compensation, front-wall alignment

def test_strafe_compensation_removes_the_systematic_shortfall():
    wm = maze_bank.open_wall_map(5, 5)
    short = {}
    for comp in (False, True):
        s = MecanumSim(wm, (2, 1), controller={"strafe_compensation": comp},
                       noise=MecanumNoise(strafe_on=True, strafe_eff_sigma=0.0))
        assert s.eta_hat == (0.90 if comp else 1.0)
        s.step_action(Action.E, 1)
        short[comp] = cell_center_world(2, 2)[0] - s.position()[0]
    assert short[False] == pytest.approx(0.018, abs=0.003) and abs(short[True]) < 0.003


def test_strafe_compensation_is_neutral_without_strafe_loss():
    s = MecanumSim(maze_bank.open_wall_map(5, 5), (2, 1))       # strafe_on false: nothing to calibrate
    assert s.eta_hat == 1.0
    s.step_action(Action.E, 1)
    assert np.linalg.norm(s.position() - cell_center_world(2, 2)) < 0.002


@pytest.mark.parametrize("aligned", [False, True])
def test_front_alignment_corrects_the_distance_along_the_move(aligned):
    import mujoco
    s = MecanumSim(_ns_corridor(4), (3, 0), controller={"front_alignment": aligned})
    s.data.qpos[s.q_xy[1]] -= 0.020                         # 20 mm behind (south of) the centre
    mujoco.mj_forward(s.model, s.data)
    res = s.step_action(Action.N, 3)                        # toward the top wall
    along = s.position()[1] - cell_center_world(*res.cell)[1]
    assert res.cell == (0, 0) and not res.collided
    if aligned:
        assert abs(along) < 0.003 and s.n_front > 0
    else:
        assert along == pytest.approx(-0.020, abs=0.002)


def test_corrections_together_stop_the_strafe_collisions():
    """The scenario that collided with centering alone: strafes then a corridor."""
    from config import WALL_DOWN, WALL_RIGHT
    wm = maze_bank.closed_wall_map(4, 3)
    for c in range(2):
        maze_bank.set_edge(wm, 3, c, WALL_RIGHT, 1)
    maze_bank.set_edge(wm, 0, 0, WALL_RIGHT, 1)
    for c in range(3):
        for r in range(3):
            maze_bank.set_edge(wm, r, c, WALL_DOWN, 1)
    s = MecanumSim(wm, (3, 0), noise=MecanumNoise(strafe_on=True, strafe_eff_sigma=0.05), seed=2)
    results = [s.step_action(a, k) for a, k in ((Action.E, 1), (Action.E, 1), (Action.N, 2))]
    assert not any(r.collided for r in results) and results[-1].cell == (1, 2)
