"""Headless MuJoCo maze environment: one discrete action in, one discrete outcome out (m1.2).

Observations use the sensing.py conventions, so the same belief filter can read
them: 4 rays in N, E, S, W order, each a digit ``0..R`` (distance of the first
wall, 0 = the robot's own edge) or ``R + 1`` (no wall within R; sensing.py calls
this digit NONE), packed into one index with ``sensing.encode_observation``.

The sensor itself never returns "nothing" (team decision, 23.11.2026): a ray
with no wall inside the sensor range reads the maximum range, so it discretizes
to R + 1 like any far reading. Misses are wrong distances (see noise.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import BUILD_DIR
from .mjcf import CELL, SENSOR_MAX_RANGE, WALL_OFFSET, build_mjcf, cell_center_world  # noqa: F401
import config
import maze_bank
from motion_model import DIR_DELTAS, Action
from sensing import SensingParams, encode_observation


@dataclass
class StepResult:
    cell: tuple[int, int]     # (row, col) after discretization
    collided: bool            # any robot-wall contact during the action
    sim_time: float           # s of simulated time used
    timed_out: bool


def discretize_ranges(ranges: np.ndarray, R: int, wall_offset: float = WALL_OFFSET) -> np.ndarray:
    """Rangefinder readings (m) -> sensing.py ray digits.

    ``round((range - wall_offset) / CELL)`` gives the wall distance j. The guide
    uses ``floor(x + 0.5)``, which is the same thing and robust to 0.99999.
    A reading that rounds past R means no wall within R, digit R + 1 (it is not
    clipped to R). A reading below the wall offset (noise) becomes 0. Negative
    values (MuJoCo's "no hit") are treated as far, for raw readings passed in.
    """
    ranges = np.asarray(ranges, dtype=float)
    v = np.floor((ranges - wall_offset) / CELL + 0.5).astype(int)
    v = np.maximum(v, 0)
    v[(ranges < 0) | (v > R)] = R + 1
    return v


class MazeSim:
    def __init__(self, wall_map: np.ndarray, start: tuple[int, int], R: int | None = None,
                 kp: float = 4.0, v_max: float = 0.30, eps_v: float = 1e-3, eps_x: float = 0.003,
                 t_min: float = 0.05, timeout: float = 3.0, xml_path: str | None = None):
        self.wall_map = np.asarray(wall_map)
        self.R = int(config.SENSING_RANGE_R if R is None else R)
        self.sensing = SensingParams(self.R, 0.0)   # for the observation encoding only
        xml = build_mjcf(self.wall_map, start, self.R)
        if xml_path:
            with open(xml_path, "w") as f:
                f.write(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.start = (int(start[0]), int(start[1]))
        self.wall_offset = WALL_OFFSET
        self.kp, self.v_max, self.eps_v, self.eps_x = kp, v_max, eps_v, eps_x
        self.t_min, self.timeout = t_min, timeout
        self.on_step = None          # optional f(data) after every physics step (viewer, video)
        m = self.model
        self.robot_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.is_wall = np.array([n.startswith(("wall_", "post_")) for n in names])
        self.rf_ids = [i for i in range(m.nsensor) if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER]
        mujoco.mj_forward(m, self.data)

    @classmethod
    def from_maze(cls, name: str, start: tuple[int, int] | None = None, write_xml: bool = True, **kw) -> "MazeSim":
        """Build from a maze in the bank (m1.0: the same maze files as the Python simulator).

        The default start is (0, 0). With ``write_xml`` the model is also saved to
        ``build/maze_<name>.xml`` so it can be opened with ``python -m mujoco.viewer``.
        """
        spec = maze_bank.load_maze(name)
        xml_path = None
        if write_xml:
            BUILD_DIR.mkdir(exist_ok=True)
            xml_path = str(BUILD_DIR / f"maze_{spec.name.replace('/', '_')}.xml")
        sim = cls(spec.wall_map, start or (0, 0), xml_path=xml_path, **kw)
        sim.goal_cells = frozenset(spec.goal_cells)
        return sim

    def _tick(self) -> None:
        if self.on_step is not None:
            self.on_step(self.data)

    # ---------- state ----------
    def cell(self) -> tuple[int, int]:
        """Discretize: displacement from the start cell centre, rounded to whole cells.
        Slide-joint qpos is relative to the body's start pos, not absolute."""
        dcol = int(round(self.data.qpos[0] / CELL))
        drow = -int(round(self.data.qpos[1] / CELL))
        return self.start[0] + drow, self.start[1] + dcol

    def teleport(self, cell: tuple[int, int]) -> None:
        """Place the robot at a cell centre at rest (used by m3 and for kidnapping)."""
        self.data.qpos[0] = (cell[1] - self.start[1]) * CELL
        self.data.qpos[1] = -(cell[0] - self.start[0]) * CELL
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def in_wall_contact(self) -> bool:
        n = self.data.ncon
        if n == 0:
            return False
        g = self.data.contact.geom[:n]            # (ncon, 2) array of geom ids
        hit = ((g[:, 0] == self.robot_geom) & self.is_wall[g[:, 1]]) | \
              ((g[:, 1] == self.robot_geom) & self.is_wall[g[:, 0]])
        return bool(hit.any())

    # ---------- action ----------
    def step_action(self, action: Action, k: int = 1) -> StepResult:
        """Drive k cells in the direction of a motion_model ``Action`` (N, NE, ...)."""
        action = Action(action)
        if not action.is_move:
            raise ValueError("DECLARE is not a motion")
        return self.step(DIR_DELTAS[int(action)], k)

    def step(self, d: tuple[int, int], k: int) -> StepResult:
        """Drive k cells in grid direction d = (drow, dcol), e.g. N = (-1, 0), NE = (-1, 1).

        The m1 controller closes the loop on the TRUE qpos, so with no noise a free
        move always lands exactly on target. m2 (noisy_env.py) closes it on odometry.
        """
        m, data = self.model, self.data
        r, c = self.cell()
        target = np.array([(c + k * d[1] - self.start[1]) * CELL,
                           -(r + k * d[0] - self.start[0]) * CELL])
        t0, collided = data.time, False
        while True:
            err = target - data.qpos[:2]
            data.ctrl[:] = np.clip(self.kp * err, -self.v_max, self.v_max)
            mujoco.mj_step(m, data)
            self._tick()
            collided |= self.in_wall_contact()
            t = data.time - t0
            settled = np.linalg.norm(data.qvel[:2]) < self.eps_v and t > self.t_min
            if settled and (np.linalg.norm(err) < self.eps_x or collided):
                break
            if t > self.timeout:
                break
        data.ctrl[:] = 0
        result = StepResult(self.cell(), collided, data.time - t0, t > self.timeout)
        if collided:
            self.recenter()
        return result

    def recenter(self) -> None:
        """After a bump the robot is pressed against a wall. Drive back to the centre of
        the current cell so the next action does not start in contact (which would be
        counted as a fresh collision)."""
        r, c = self.cell()
        target = np.array([(c - self.start[1]) * CELL, -(r - self.start[0]) * CELL])
        t0 = self.data.time
        while self.data.time - t0 < self.timeout:
            err = target - self.data.qpos[:2]
            self.data.ctrl[:] = np.clip(self.kp * err, -self.v_max, self.v_max)
            mujoco.mj_step(self.model, self.data)
            self._tick()
            if np.linalg.norm(err) < self.eps_x and np.linalg.norm(self.data.qvel[:2]) < self.eps_v:
                break
        self.data.ctrl[:] = 0

    # ---------- observation ----------
    def raw_ranges(self) -> np.ndarray:
        """MuJoCo rangefinder values in m, N, E, S, W; negative means no hit within the cutoff."""
        return np.array([self.data.sensordata[self.model.sensor_adr[i]] for i in self.rf_ids])

    def ranges(self) -> np.ndarray:
        """Noise-free sensor readings in m, N, E, S, W. Like the real ToF sensor, a ray
        never reports 'nothing': with no wall within range it reads the maximum range."""
        r = self.raw_ranges()
        return np.where(r < 0, SENSOR_MAX_RANGE, r)

    def true_digits(self) -> np.ndarray:
        """Noise-free ray digits (N, E, S, W) from the raw readings."""
        return discretize_ranges(self.ranges(), self.R, self.wall_offset)

    def observe(self) -> np.ndarray:
        """Ray digits (N, E, S, W): 0..R, or R + 1 for no wall within R. m2 adds noise."""
        return self.true_digits()

    def observe_index(self) -> int:
        """The observation as a sensing.py index (what the belief filter and policies read)."""
        return encode_observation(tuple(int(v) for v in self.observe()), self.sensing)
