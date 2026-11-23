"""Mecanum robot, Level A: planar chassis, kinematic wheel coupling (design proposal 23.11.2026).

What is modelled (proposal §4, §5, §7, and the team's answers of 23.11.2026):

- A 110 x 110 mm square chassis, 300 g. It slides in x and y; its box collides
  with walls but never touches the floor.
- Yaw (D1): ``yaw_mode="locked"`` (default) keeps the heading perfect.
  ``"held"`` adds a yaw hinge, an IMU (MuJoCo gyro + accelerometer with white
  noise and a random-walk gyro bias), a yaw-rate disturbance, and a heading-hold
  loop on the integrated gyro.
- Start heading (D2): the robot always faces maze north when placed.
- Four wheels in X configuration (FL, FR, RL, RR), each a hinge driven by an
  N20 50:1 gearmotor model: torque = stall * duty - (stall / omega0) * omega,
  MuJoCo's ``general`` actuator with an affine bias. Wheels are visual only.
- Level A coupling: every physics step, the actual wheel speeds go through
  mecanum forward kinematics (with per-wheel radius errors, strafe efficiency
  and per-action slip from ``MecanumNoise``) to a body twist, which stiff
  velocity servos on the planar joints track, force-limited at the traction
  limit. A wall still pushes back, so a blocked robot stops while its wheels
  keep turning and its encoders count travel that never happened.
- Encoders: 12 CPR at the motor x 51.45 = 617.4 counts per wheel turn.
- Controller: trapezoidal profile from the believed cell centre to the target
  centre; position loop on ENCODER ODOMETRY (rotated by the gyro heading in held
  mode; sideways part scaled by the calibrated strafe efficiency); wall centering:
  at the ToF rate, side sensors that see a wall in the robot's own cell correct
  the lateral odometry toward 50 mm from that wall; front-wall alignment: near
  the end of a move, the sensor facing a wall in the target cell corrects the
  distance along the move; heading hold (held mode); mecanum IK; per-wheel PI
  with feedforward on encoder speed. All three corrections are on by default.
- Sensors: four VL53L1X time-of-flight sensors, see tof.py.

Frames: body x forward, y left, z up. At heading 0 the robot faces maze north
(world +y), so action N drives forward and E strafes right. The controller works
in the "nominal" frame, the body frame at heading 0.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from .env import StepResult, discretize_ranges
from .mjcf import CELL, WALL_DEFAULT, WALL_T, cell_center_world, maze_worldbody
from .tof import TofConfig, TofModel
import config
from motion_model import DIR_DELTAS, Action
from sensing import SensingParams, encode_observation

WHEELS = ("FL", "FR", "RL", "RR")
_WHEEL_SIGNS = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}   # (x, y) in body frame
YAW_MODES = ("locked", "held")
# Side sensors (index into N, E, S, W) and the lateral axis they measure, per move axis.
# For a move along body x the lateral axis is +y (W sensor on +y, E sensor on -y), etc.
_SIDES = {0: (np.array([0.0, 1.0]), 3, 1), 1: (np.array([1.0, 0.0]), 0, 2)}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotConfig:
    """Physical robot (proposal §4). 'Brief' / 'sourced' / *assumed* as in the proposal."""

    footprint: float = 0.110          # m, square incl. wheels (brief)
    chassis_h: float = 0.040          # m (brief)
    chassis_z0: float = 0.005         # m, ground clearance of the chassis box (assumed)
    wheel_r: float = 0.019            # m, 38 mm wheels (brief)
    wheel_w: float = 0.020            # m (brief)
    half_track: float = 0.045         # m, lateral wheel offset (l_x)
    half_base: float = 0.036          # m, longitudinal wheel offset (l_y)
    m_chassis: float = 0.218          # kg, 180 g chassis + electronics + 4 x 9.5 g motors
    m_battery: float = 0.050          # kg (assumed)
    battery_size: tuple[float, float, float] = (0.070, 0.035, 0.020)   # m (assumed)
    battery_x: float = -0.010         # m, centred 10 mm behind the middle (assumed)
    m_wheel: float = 0.008            # kg (assumed)
    wheel_armature: float = 1.0e-6    # kg m^2, reflected rotor inertia (assumed)
    gear: float = 51.45               # (sourced)
    no_load_rpm: float = 590.0        # at the wheel, 6 V (sourced)
    stall_torque: float = 0.084       # N m at the wheel, 6 V (sourced)
    cpr_motor: float = 12.0           # encoder counts per motor turn (sourced)
    tof_offset: float = 0.035         # m, sensor distance from the centre (brief)
    tof_z: float = 0.025              # m, sensor height (brief)
    traction_force: float = 2.35      # N, mu * m * g with mu = 0.8: planar servo force limit (assumed)
    traction_torque: float = 0.14     # N m, yaw servo limit, traction_force x 60 mm (assumed)
    servo_kv: float = 60.0            # N s / m, planar velocity servos (stiff)
    yaw_kv: float = 0.1               # N m s / rad, yaw velocity servo (held mode)
    timestep: float = 0.001           # s

    @property
    def counts_per_rev(self) -> float:
        return self.cpr_motor * self.gear                    # 617.4

    @property
    def omega0(self) -> float:
        return self.no_load_rpm * 2 * np.pi / 60.0           # rad/s

    @property
    def L(self) -> float:
        return self.half_track + self.half_base

    @property
    def tof_d0(self) -> float:
        """Reading at a cell centre facing an adjacent wall: 90 - 5 - 35 = 50 mm."""
        return CELL / 2 - WALL_T / 2 - self.tof_offset


@dataclass(frozen=True)
class MecanumController:
    yaw_mode: str = "locked"     # locked | held (D1)
    v_max: float = 0.30          # m/s cruise (assumed, about 25% of top speed)
    accel: float = 1.0           # m/s^2 (assumed)
    kp_pos: float = 5.0          # 1/s, position loop on odometry
    kp_heading: float = 10.0     # 1/s, heading hold on the integrated gyro (held mode)
    wall_centering: bool = True  # side ToF corrects lateral odometry (team decision)
    centering_gain: float = 0.5  # fraction of the lateral error corrected per ToF sample
    centering_window: float = 0.06  # m, use side readings only this close to a cell centre, where
                                    # the side cone lies fully on the own cell's wall (not a gap)
    front_alignment: bool = True    # front ToF facing a wall in the target cell corrects the distance
                                    # along the move near its end (team decision)
    strafe_compensation: bool = True  # firmware scales sideways odometry by the robot's calibrated
                                      # mean strafe efficiency (team decision)
    tof_period: float = 0.033    # s, ToF timing budget (30 Hz)
    kp_wheel: float = 0.005      # duty per rad/s
    ki_wheel: float = 0.05       # duty per rad
    speed_window: int = 5        # physics steps for the encoder speed estimate
    eps_x: float = 0.002         # m, odometry error to stop
    eps_v: float = 0.005         # m/s, speed to stop
    t_min: float = 0.05          # s
    timeout: float = 3.0         # s
    settle: float = 0.02         # s with all commands zero after a move

    def __post_init__(self):
        if self.yaw_mode not in YAW_MODES:
            raise ValueError(f"yaw_mode must be one of {YAW_MODES}, got {self.yaw_mode!r}")


@dataclass(frozen=True)
class MecanumNoise:
    """Level A slip model (proposal §5), IMU, and ToF noise (§6, decision 2). Placeholders."""

    radius_on: bool = False
    radius_sigma: float = 0.01        # effective wheel-radius error per wheel, drawn per run
    strafe_on: bool = False
    strafe_eff_mean: float = 0.90     # sideways moves come up short, drawn per run
    strafe_eff_sigma: float = 0.05
    slip_on: bool = False
    slip_sigma: float = 0.02          # per action, on each twist component
    yaw_drift_on: bool = False        # held mode only
    yaw_drift_rw: float = 0.01        # rad/s per sqrt(s), random walk of a yaw-rate disturbance
    imu_on: bool = False              # held mode only
    gyro_noise: float = 0.005         # rad/s, white, per sample
    gyro_bias_rw: float = 1.0e-4      # rad/s per sqrt(s)
    accel_noise: float = 0.02         # m/s^2
    tof_noise_on: bool = False
    tof_a: float = 0.003              # m, sigma(d) = a + b * d
    tof_b: float = 0.01
    tof_spread_gain: float = 1.0      # widen sigma by the spread of the cone's returns
    tof_close_factor: float = 5.0     # sigma x this below the 40 mm minimum range
    bad_return_on: bool = False
    p_bad_return: float = 0.05        # decision 2: a miss is a noisy reading, never empty
    bad_return_sigma: float = 0.15    # m

    @classmethod
    def all_on(cls, **overrides) -> "MecanumNoise":
        flags = {f.name: True for f in dataclasses.fields(cls) if f.name.endswith("_on")}
        return cls(**{**flags, **overrides})


# --------------------------------------------------------------------------
# Kinematics (Taheri et al. 2015, X configuration; wheels FL, FR, RL, RR)
# --------------------------------------------------------------------------

def ik(twist, r: float, L: float) -> np.ndarray:
    """Body twist (vx, vy, wz) -> wheel speeds (rad/s)."""
    vx, vy, wz = twist
    return np.array([vx - vy - L * wz, vx + vy + L * wz, vx + vy - L * wz, vx - vy + L * wz]) / r


def fk(w, r, L: float) -> np.ndarray:
    """Wheel speeds (or angle increments) -> body twist (or displacement). ``r`` may be
    one radius or four (per-wheel effective radii)."""
    v = np.asarray(w, dtype=float) * np.asarray(r, dtype=float)
    return np.array([v.sum() / 4, (-v[0] + v[1] + v[2] - v[3]) / 4, (-v[0] + v[1] - v[2] + v[3]) / (4 * L)])


def rot(v, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def body_to_world(v_body, yaw: float = 0.0) -> np.ndarray:
    """Body frame -> world. At yaw 0 the body x axis is world +y (maze north)."""
    return rot(v_body, np.pi / 2 + yaw)


def grid_to_body(d: tuple[int, int]) -> np.ndarray:
    """Grid step (drow, dcol) -> nominal-frame direction (x forward = north, y left = west)."""
    return np.array([-d[0], -d[1]], dtype=float)


class Trapezoid:
    """Trapezoidal (or triangular) velocity profile over distance D."""

    def __init__(self, D: float, v_max: float, accel: float):
        self.D, self.a = D, accel
        if D * accel < v_max ** 2:                       # never reaches v_max
            self.v = np.sqrt(D * accel)
            self.t_a, self.t_c = self.v / accel, 0.0
        else:
            self.v = v_max
            self.t_a, self.t_c = v_max / accel, (D - v_max ** 2 / accel) / v_max
        self.T = 2 * self.t_a + self.t_c

    def __call__(self, t: float) -> tuple[float, float]:
        a, v, t_a, t_c = self.a, self.v, self.t_a, self.t_c
        if t <= 0:
            return 0.0, 0.0
        if t < t_a:
            return 0.5 * a * t * t, a * t
        if t < t_a + t_c:
            return 0.5 * a * t_a ** 2 + v * (t - t_a), v
        if t < self.T:
            tr = self.T - t
            return self.D - 0.5 * a * tr * tr, a * tr
        return self.D, 0.0


# --------------------------------------------------------------------------
# MJCF
# --------------------------------------------------------------------------

def robot_xml(rc: RobotConfig, yaw_mode: str = "locked") -> tuple[str, str, str]:
    """(worldbody robot, actuators, sensors). The robot body sits at the world origin so
    the slide joints carry the absolute position; in held mode a yaw hinge follows them."""
    half = rc.footprint / 2
    zc = rc.chassis_z0 + rc.chassis_h / 2
    bx, by, bz = rc.battery_size
    wheels = []
    for name in WHEELS:
        sx, sy = _WHEEL_SIGNS[name]
        wheels.append(
            f'<body name="wheel_{name}" pos="{sx * rc.half_base:.4f} {sy * rc.half_track:.4f} {rc.wheel_r:.4f}">\n'
            f'          <joint name="w_{name}" type="hinge" axis="0 1 0" armature="{rc.wheel_armature:g}"/>\n'
            f'          <geom name="wheel_{name}_geom" type="cylinder" size="{rc.wheel_r} {rc.wheel_w / 2}" '
            f'zaxis="0 1 0" mass="{rc.m_wheel}" rgba=".1 .1 .1 1" contype="0" conaffinity="0" group="1"/>\n'
            f'        </body>')
    sites = "\n        ".join(
        f'<site name="tof_{k}" pos="{dx * rc.tof_offset:.4f} {dy * rc.tof_offset:.4f} {rc.tof_z}" size="0.004" rgba="1 .8 0 1" group="1"/>'
        for k, (dx, dy) in zip(("N", "E", "S", "W"), ((1, 0), (0, -1), (-1, 0), (0, 1))))
    yaw_joint = '<joint name="hinge_yaw" type="hinge" axis="0 0 1"/>' if yaw_mode == "held" else ""
    body = f"""<body name="robot" pos="0 0 0">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      {yaw_joint}
      <body name="chassis" quat="0.70710678 0 0 0.70710678">
        <geom name="chassis_geom" type="box" size="{half} {half} {rc.chassis_h / 2}" pos="0 0 {zc:.4f}"
              mass="{rc.m_chassis}" rgba=".2 .55 1 .45" contype="1" conaffinity="1" group="1"/>
        <geom name="battery" type="box" size="{bx / 2} {by / 2} {bz / 2}" pos="{rc.battery_x} 0 {rc.chassis_z0 + bz / 2:.4f}"
              mass="{rc.m_battery}" rgba=".9 .3 .2 1" contype="0" conaffinity="0" group="1"/>
        <geom name="front_marker" type="box" size="0.006 0.02 0.002" pos="{half - 0.008:.4f} 0 {rc.chassis_z0 + rc.chassis_h + 0.002:.4f}"
              rgba="1 1 1 1" contype="0" conaffinity="0" mass="0" group="1"/>
        <site name="imu" pos="0 0 {zc:.4f}" size="0.004" group="1"/>
        {sites}
        {chr(10).join("        " + w for w in wheels)}
      </body>
    </body>"""
    b = rc.stall_torque / rc.omega0
    motors = "\n    ".join(
        f'<general name="m_{n}" joint="w_{n}" gainprm="{rc.stall_torque}" biastype="affine" '
        f'biasprm="0 0 {-b:.6g}" ctrllimited="true" ctrlrange="-1 1"/>' for n in WHEELS)
    f = rc.traction_force
    yaw_servo = (f'<velocity name="servo_yaw" joint="hinge_yaw" kv="{rc.yaw_kv}" forcelimited="true" '
                 f'forcerange="-{rc.traction_torque} {rc.traction_torque}"/>' if yaw_mode == "held" else "")
    actuators = f"""<velocity name="servo_x" joint="slide_x" kv="{rc.servo_kv}" forcelimited="true" forcerange="-{f} {f}"/>
    <velocity name="servo_y" joint="slide_y" kv="{rc.servo_kv}" forcelimited="true" forcerange="-{f} {f}"/>
    {yaw_servo}
    {motors}"""
    sensors = """<gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>"""
    return body, actuators, sensors


def build_mecanum_mjcf(wall_map: np.ndarray, rc: RobotConfig = RobotConfig(), yaw_mode: str = "locked") -> str:
    wall_map = np.asarray(wall_map)
    rows, cols = wall_map.shape[:2]
    body, actuators, sensors = robot_xml(rc, yaw_mode)
    return f"""<mujoco model="mecanum_maze_{rows}x{cols}">
  <option timestep="{rc.timestep}" integrator="implicitfast"/>
  <default>
    {WALL_DEFAULT}
  </default>
  <worldbody>
    {maze_worldbody(wall_map)}
    {body}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
  <sensor>
    {sensors}
  </sensor>
</mujoco>
"""


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class MecanumSim:
    """Same interface as MazeSim / NoisyMazeSim: teleport, step(d, k), step_action,
    cell, ranges, true_digits, observe, observe_index, new_run, rng.

    ``on_step``: optional callback ``f(data)`` after every physics step (viewer, video)."""

    def __init__(self, wall_map: np.ndarray, start: tuple[int, int] = (0, 0), R: int | None = None,
                 robot: RobotConfig | None = None, controller: MecanumController | dict | None = None,
                 noise: MecanumNoise | dict | None = None, tof: TofConfig | dict | None = None,
                 seed: int = 0, xml_path: str | None = None):
        self.wall_map = np.asarray(wall_map)
        self.rows, self.cols = self.wall_map.shape[:2]
        self.R = int(config.SENSING_RANGE_R if R is None else R)
        self.sensing = SensingParams(self.R, 0.0)
        self.rc = robot or RobotConfig()
        self.ctrl = controller if isinstance(controller, MecanumController) else MecanumController(**(controller or {}))
        self.noise = noise if isinstance(noise, MecanumNoise) else MecanumNoise(**(noise or {}))
        tof_cfg = tof if isinstance(tof, TofConfig) else TofConfig(**(tof or {}))
        self.held = self.ctrl.yaw_mode == "held"
        xml = build_mecanum_mjcf(self.wall_map, self.rc, self.ctrl.yaw_mode)
        if xml_path:
            with open(xml_path, "w") as f:
                f.write(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.q_xy = np.array([m.jnt_qposadr[jid("slide_x")], m.jnt_qposadr[jid("slide_y")]])
        self.q_yaw = int(m.jnt_qposadr[jid("hinge_yaw")]) if self.held else None
        self.q_wheel = np.array([m.jnt_qposadr[jid(f"w_{n}")] for n in WHEELS])
        self.v_wheel = np.array([m.jnt_dofadr[jid(f"w_{n}")] for n in WHEELS])
        self.a_servo = np.array([aid("servo_x"), aid("servo_y")])
        self.a_yaw = aid("servo_yaw") if self.held else None
        self.a_motor = np.array([aid(f"m_{n}") for n in WHEELS])
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
        self.s_gyro = int(m.sensor_adr[sid("gyro")])
        self.s_accel = int(m.sensor_adr[sid("accel")])
        self.chassis_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "chassis_geom")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.is_wall = np.array([n.startswith("wall_") for n in names])
        self.tof = TofModel(m, tof_cfg, self.rc.tof_offset, self.rc.tof_z, names)
        self.wall_offset = self.rc.tof_d0
        self.rng = np.random.default_rng(seed)
        self.radii = np.full(4, self.rc.wheel_r)
        self.eta_y = 1.0
        self.on_step: Callable[[mujoco.MjData], None] | None = None
        self.psi_hat = 0.0          # integrated gyro heading (held mode)
        self.gyro_bias = 0.0
        self.yaw_dist = 0.0         # yaw-rate disturbance (held mode)
        self.n_centering = 0        # wall-centering corrections applied (diagnostics)
        self.n_front = 0            # front-wall alignments applied (diagnostics)
        self.eta_hat = 1.0
        self.new_run()
        self.teleport(start)

    # ---------- per-run noise ----------
    def new_run(self) -> None:
        """Per-run draws: effective wheel radii and strafe efficiency."""
        nz, r = self.noise, self.rc.wheel_r
        self.radii = r * (1 + self.rng.normal(0, nz.radius_sigma, 4)) if nz.radius_on else np.full(4, r)
        self.eta_y = float(self.rng.normal(nz.strafe_eff_mean, nz.strafe_eff_sigma)) if nz.strafe_on else 1.0
        # What the firmware believes: the robot's calibrated MEAN strafe efficiency, not this
        # run's draw (1.0 when the model has no strafe loss, i.e. nothing to calibrate).
        self.eta_hat = nz.strafe_eff_mean if (self.ctrl.strafe_compensation and nz.strafe_on) else 1.0

    # ---------- state ----------
    def position(self) -> np.ndarray:
        return self.data.qpos[self.q_xy].copy()

    def yaw(self) -> float:
        """True heading error (rad, counter-clockwise); always 0 in locked mode."""
        return float(self.data.qpos[self.q_yaw]) if self.held else 0.0

    def cell(self) -> tuple[int, int]:
        """The cell containing the robot centre (absolute world coordinates, so floor)."""
        x, y = self.position()
        return int(np.floor(-y / CELL)), int(np.floor(x / CELL))

    def teleport(self, cell: tuple[int, int]) -> None:
        """Place the robot at a cell centre, at rest, facing north (D2)."""
        x, y = cell_center_world(*cell)
        self.data.qpos[self.q_xy] = (x, y)
        if self.held:
            self.data.qpos[self.q_yaw] = 0.0
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.psi_hat, self.gyro_bias, self.yaw_dist = 0.0, 0.0, 0.0
        mujoco.mj_forward(self.model, self.data)

    def encoder_counts(self) -> np.ndarray:
        return np.floor(self.data.qpos[self.q_wheel] / (2 * np.pi) * self.rc.counts_per_rev)

    def imu(self) -> tuple[float, np.ndarray]:
        """(measured yaw rate, measured acceleration) with IMU noise when on."""
        gz = float(self.data.sensordata[self.s_gyro + 2])
        acc = self.data.sensordata[self.s_accel:self.s_accel + 3].copy()
        if self.noise.imu_on:
            gz += self.gyro_bias + self.rng.normal(0.0, self.noise.gyro_noise)
            acc += self.rng.normal(0.0, self.noise.accel_noise, 3)
        return gz, acc

    def in_wall_contact(self) -> bool:
        n = self.data.ncon
        if n == 0:
            return False
        g = self.data.contact.geom[:n]
        hit = ((g[:, 0] == self.chassis_geom) & self.is_wall[g[:, 1]]) | \
              ((g[:, 1] == self.chassis_geom) & self.is_wall[g[:, 0]])
        return bool(hit.any())

    # ---------- action ----------
    def step_action(self, action: Action, k: int = 1) -> StepResult:
        action = Action(action)
        if not action.is_move:
            raise ValueError("DECLARE is not a motion")
        return self.step(DIR_DELTAS[int(action)], k)

    _GATE = 0.045   # m: a reading within this of 50 mm is a wall in the own cell, plausibly placed

    def _tof_noisy(self) -> np.ndarray:
        """One ToF sample as the firmware sees it (with sensor noise when on)."""
        readings, spread = self.tof.read(self.data, self.position(), self.yaw())
        return self.tof.corrupt(readings, spread, self.noise, self.rng)

    def _lateral_fix(self, u: np.ndarray, noisy: np.ndarray) -> float | None:
        """Lateral offset (m, along the side axis) measured by the side ToF sensors that
        see a wall in the robot's own cell, or None. Cardinal moves only."""
        axis = int(np.argmax(np.abs(u)))
        _, plus_i, minus_i = _SIDES[axis]
        d0, est = self.wall_offset, []
        if abs(noisy[plus_i] - d0) < self._GATE:
            est.append(d0 - noisy[plus_i])                      # closer to the + wall: displaced toward +
        if abs(noisy[minus_i] - d0) < self._GATE:
            est.append(noisy[minus_i] - d0)
        return float(np.mean(est)) if est else None

    def _front_fix(self, u: np.ndarray, noisy: np.ndarray) -> float | None:
        """Remaining distance (m) to the target centre, measured by the sensor facing a wall
        on the far edge of the target cell, or None. Cardinal moves only."""
        axis = int(np.argmax(np.abs(u)))
        front_i = (0 if u[0] > 0 else 2) if axis == 0 else (3 if u[1] > 0 else 1)   # N, S | W, E
        r = noisy[front_i] - self.wall_offset
        return float(r) if abs(r) < self._GATE else None

    def step(self, d: tuple[int, int], k: int) -> StepResult:
        """Drive k cells in grid direction d. The robot believes it starts at a cell centre
        and only knows its encoders, its gyro (held mode) and its ToF sensors."""
        m, data, rc, cc, nz = self.model, self.data, self.rc, self.ctrl, self.noise
        dt, r, L = m.opt.timestep, rc.wheel_r, rc.L
        u = grid_to_body(d)
        dist = k * CELL * float(np.linalg.norm(u))
        u /= np.linalg.norm(u)
        goal = dist * u
        axis = int(np.argmax(np.abs(u)))
        side = _SIDES[axis][0] if abs(u[1 - axis]) < 1e-9 else None
        profile = Trapezoid(dist, cc.v_max, cc.accel)
        slip = 1 + self.rng.normal(0, nz.slip_sigma, 3) if nz.slip_on else np.ones(3)
        rad_per_count = 2 * np.pi / rc.counts_per_rev
        tof_every = max(1, int(round(cc.tof_period / dt)))

        counts_prev = self.encoder_counts()
        hist = collections.deque([counts_prev], maxlen=cc.speed_window + 1)
        odo = np.zeros(2)                                       # nominal frame
        integ = np.zeros(4)
        t0, n, collided, timed_out = data.time, 0, False, False
        while True:
            t = data.time - t0
            counts = self.encoder_counts()
            d_body = fk((counts - counts_prev) * rad_per_count, r, L)[:2] * (1.0, self.eta_hat)
            odo += rot(d_body, self.psi_hat)
            counts_prev = counts
            hist.append(counts)
            w_hat = (hist[-1] - hist[0]) * rad_per_count / (max(len(hist) - 1, 1) * dt)

            if side is not None and n > 0 and n % tof_every == 0 and (cc.wall_centering or cc.front_alignment):
                along = odo @ u                                 # believed distance travelled
                near_centre = min(along % CELL, CELL - along % CELL) < cc.centering_window
                near_end = dist - along < cc.centering_window
                if (cc.wall_centering and near_centre) or (cc.front_alignment and near_end):
                    noisy = self._tof_noisy()
                    fix = self._lateral_fix(u, noisy) if cc.wall_centering and near_centre else None
                    if fix is not None:
                        odo += cc.centering_gain * (fix - odo @ side) * side
                        self.n_centering += 1
                    fix = self._front_fix(u, noisy) if cc.front_alignment and near_end else None
                    if fix is not None:
                        odo += cc.centering_gain * ((dist - fix) - odo @ u) * u
                        self.n_front += 1

            s_ref, v_ref = profile(t)
            v_nom = v_ref * u + cc.kp_pos * (s_ref * u - odo)
            v_cmd = rot(v_nom, -self.psi_hat)
            wz_cmd = -cc.kp_heading * self.psi_hat if self.held else 0.0
            w_sp = ik((v_cmd[0], v_cmd[1], wz_cmd), r, L)
            err_w = w_sp - w_hat
            duty_raw = w_sp / rc.omega0 + cc.kp_wheel * err_w + cc.ki_wheel * integ
            duty = np.clip(duty_raw, -1.0, 1.0)
            integ += np.where(np.abs(duty_raw) < 1.0, err_w * dt, 0.0)     # anti-windup
            data.ctrl[self.a_motor] = duty

            # Level A coupling: actual wheel speeds -> body twist -> planar (and yaw) servos
            v_body = fk(data.qvel[self.v_wheel], self.radii, L)
            v_body[1] *= self.eta_y
            v_body *= slip
            data.ctrl[self.a_servo] = body_to_world(v_body[:2], self.yaw())
            if self.held:
                if nz.yaw_drift_on:
                    self.yaw_dist += self.rng.normal(0.0, nz.yaw_drift_rw * np.sqrt(dt))
                data.ctrl[self.a_yaw] = v_body[2] + self.yaw_dist

            mujoco.mj_step(m, data)
            n += 1
            if self.held:
                if nz.imu_on:
                    self.gyro_bias += self.rng.normal(0.0, nz.gyro_bias_rw * np.sqrt(dt))
                self.psi_hat += self.imu()[0] * dt
            if self.on_step is not None:
                self.on_step(data)
            collided |= self.in_wall_contact()
            speed = float(np.linalg.norm(fk(w_hat, r, L)[:2]))
            if t > cc.t_min and t >= profile.T and np.linalg.norm(goal - odo) < cc.eps_x and speed < cc.eps_v:
                break
            if t > cc.timeout:
                timed_out = True
                break
        data.ctrl[:] = 0
        for _ in range(int(round(cc.settle / dt))):
            mujoco.mj_step(m, data)
            if self.on_step is not None:
                self.on_step(data)
            collided |= self.in_wall_contact()
        return StepResult(self.cell(), collided, data.time - t0, timed_out)

    # ---------- observation ----------
    def ranges(self) -> np.ndarray:
        """Noise-free ToF readings (m), N, E, S, W. Never empty (decision 2)."""
        return self.tof.read(self.data, self.position(), self.yaw())[0]

    def true_digits(self) -> np.ndarray:
        return discretize_ranges(self.ranges(), self.R, self.wall_offset)

    def observe(self) -> np.ndarray:
        readings, spread = self.tof.read(self.data, self.position(), self.yaw())
        noisy = self.tof.corrupt(readings, spread, self.noise, self.rng)
        return discretize_ranges(noisy, self.R, self.wall_offset)

    def observe_index(self) -> int:
        return encode_observation(tuple(int(v) for v in self.observe()), self.sensing)


def mecanum_config_fields() -> dict[str, set[str]]:
    """Valid keys of the YAML ``mecanum`` section, for mj_config validation."""
    return {"controller": {f.name for f in dataclasses.fields(MecanumController)},
            "noise": {f.name for f in dataclasses.fields(MecanumNoise)},
            "tof": {f.name for f in dataclasses.fields(TofConfig)}}
