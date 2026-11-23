"""MazeSim + m2 noise. The controller closes the loop on ODOMETRY, not on true qpos."""

from __future__ import annotations

import mujoco
import numpy as np

from .env import MazeSim, StepResult, discretize_ranges
from .mjcf import CELL, SENSOR_MAX_RANGE, SENSOR_MIN_RANGE
from .noise import (NoiseConfig, corrupt_ranges, sample_friction, sample_stop_jitter,
                    sample_vel_gain)


class NoisyMazeSim(MazeSim):
    def __init__(self, *args, noise: NoiseConfig | None = None, seed: int = 0, **kw):
        super().__init__(*args, **kw)
        self.noise = noise or NoiseConfig.all_off()
        self.rng = np.random.default_rng(seed)
        self.new_run()

    def new_run(self) -> None:
        """Per-run draws (friction). Call once per episode / calibration batch.
        Geom friction does nothing here (the robot never touches the floor), so
        friction is joint frictionloss, which can change without a recompile."""
        self.model.dof_frictionloss[:] = sample_friction(self.noise, self.rng)

    def step(self, d: tuple[int, int], k: int) -> StepResult:
        m, data = self.model, self.data
        goal = np.array([k * d[1], -k * d[0]], float) * CELL   # displacement the robot BELIEVES it makes
        odo = np.zeros(2)                                       # encoder-style odometry
        gain = sample_vel_gain(self.noise, self.rng)            # per-action 2x2 command error
        t0, collided = data.time, False
        while True:
            err = goal - odo
            v_cmd = np.clip(self.kp * err, -self.v_max, self.v_max)
            data.ctrl[:] = gain @ v_cmd              # what the motors actually get
            mujoco.mj_step(m, data)
            odo += v_cmd * m.opt.timestep            # odometry integrates the NOMINAL command
            collided |= self.in_wall_contact()
            t = data.time - t0
            if np.linalg.norm(err) < self.eps_x and t > self.t_min:
                break
            if collided and np.linalg.norm(data.qvel[:2]) < self.eps_v and t > self.t_min:
                break
            if t > self.timeout:
                break
        jitter_end = data.time + sample_stop_jitter(self.noise, self.rng)
        while data.time < jitter_end:                # late stop: keep the last command
            mujoco.mj_step(m, data)
            collided |= self.in_wall_contact()
        data.ctrl[:] = 0
        for _ in range(int(0.2 / m.opt.timestep)):   # let it come to rest before discretizing
            mujoco.mj_step(m, data)
            collided |= self.in_wall_contact()
        res = StepResult(self.cell(), collided, data.time - t0, t > self.timeout)
        if collided:
            self.recenter()
        return res

    def observe(self) -> np.ndarray:
        """Ray digits with m2 range noise and bad returns (the MJCF ``noise`` attribute is
        ignored by MuJoCo, so all sensor noise is added here)."""
        r = corrupt_ranges(self.ranges(), self.noise, self.rng, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE)
        return discretize_ranges(r, self.R, self.wall_offset)
