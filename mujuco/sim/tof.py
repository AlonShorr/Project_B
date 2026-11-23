"""VL53L1X time-of-flight model for the mecanum robot (design proposal §6, team decision 2).

Each of the four sensors (N, E, S, W in the body frame, 35 mm from the centre,
25 mm up) casts a grid of rays inside its cone with one ``mj_multiRay`` call.
Only geom group 0 (walls and floor) is visible, so the robot never sees itself.

Return strength per ray:  w_i = b(alpha_i) * rho_i * cos(theta_i) / d_i^2
  b      Gaussian beam profile over the off-axis angle (sigma = beam_sigma_frac * half-FoV, assumed)
  rho    reflectance of the geom hit (mjcf.REFLECTANCE: walls 0.5, floor 0.3, assumed)
  theta  incidence angle, from the hit normal
The reading is the return-weighted mean distance, as the sensor averages all
returned photons. Floor hits at grazing angles and far over-the-wall hits pull
it a little; how much is proposal question D5 (check against a real sensor).

The sensor never reports "nothing" (decision 2): with too little signal, or a
mean beyond the maximum range, it reads the maximum range, which discretizes
to "no wall within R". Misses are noisy readings (``corrupt``).

``model="center_ray"`` keeps only the axis ray (a thin beam), for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .mjcf import REFLECTANCE

# Body-frame sensor directions in observation order (sensing.py: N, E, S, W).
SENSOR_DIRS = ((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0))
_GROUP0_ONLY = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class TofConfig:
    model: str = "cone"            # cone | center_ray
    fov_deg: float = 15.0          # full cone: 15 = ROI 4x4, 27 = ROI 16x16 (sourced). With 27 the
                                   # cone edge reaches the side walls of a 170 mm corridor from about
                                   # 350 mm, and 831 of 1,705 bank cells give a wrong digit at R = 2
                                   # even with no noise; with 15, none do (proposal D5).
    grid: int = 5                  # grid x grid rays, trimmed to the cone (21 for 5)
    beam_sigma_frac: float = 0.5   # beam profile sigma as a fraction of the half-FoV (assumed)
    min_range: float = 0.040       # m, below this readings are inaccurate (sourced)
    max_range: float = 1.300       # m (brief)


def _body_to_world(v) -> np.ndarray:
    """Heading 0: body x -> world +y, body y -> world -x (as mecanum.body_to_world)."""
    return np.array([-v[1], v[0]])


class TofModel:
    def __init__(self, model: mujoco.MjModel, cfg: TofConfig, offset: float, z: float, geom_names: list[str]):
        if cfg.model not in ("cone", "center_ray"):
            raise ValueError(f"tof.model must be cone or center_ray, got {cfg.model!r}")
        self.m, self.cfg, self.z = model, cfg, z
        self.refl = np.array([REFLECTANCE["floor"] if n == "floor" else
                              REFLECTANCE["wall"] if n.startswith("wall_") else 0.0 for n in geom_names])
        half = np.deg2rad(cfg.fov_deg) / 2
        if cfg.model == "center_ray" or cfg.grid <= 1:
            ab = np.zeros((1, 2))
        else:
            g = np.linspace(-1.0, 1.0, cfg.grid)
            ab = np.array([(a, b) for b in g for a in g if a * a + b * b <= 1.25 + 1e-9])
            ab *= 1.0 / np.sqrt(1.25)                       # outermost rays on the cone edge
        ang_h, ang_v = ab[:, 0] * half, ab[:, 1] * half
        sigma_b = cfg.beam_sigma_frac * half
        self.offsets, self.dirs, self.beam = [], [], []
        for bx, by in SENSOR_DIRS:
            axis2 = _body_to_world((bx, by))
            axis = np.array([axis2[0], axis2[1], 0.0])
            side = np.array([-axis[1], axis[0], 0.0])
            up = np.array([0.0, 0.0, 1.0])
            dirs = axis + np.tan(ang_h)[:, None] * side + np.tan(ang_v)[:, None] * up
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            alpha = np.arccos(np.clip(dirs @ axis, -1.0, 1.0))
            self.offsets.append(axis2 * offset)
            self.dirs.append(dirs)
            self.beam.append(np.exp(-alpha ** 2 / (2 * sigma_b ** 2)) if sigma_b > 0 else np.ones(len(dirs)))
        self.n = len(ab)
        # signal threshold: a reflectance-0.5 wall filling the cone at the maximum range
        self.w_min = float(self.beam[0].sum() * REFLECTANCE["wall"] / cfg.max_range ** 2)
        self._geomid = np.zeros(self.n, dtype=np.int32)
        self._dist = np.zeros(self.n)
        self._normal = np.zeros(3 * self.n)

    def read(self, data: mujoco.MjData, pos_xy: np.ndarray, yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Noise-free readings (m) and the return-weighted spread (m) of each sensor.
        ``yaw`` (rad) rotates the sensors with the robot (held mode)."""
        cfg = self.cfg
        readings, spread = np.full(4, cfg.max_range), np.zeros(4)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for i in range(4):
            off = R[:2, :2] @ self.offsets[i]
            pnt = np.array([pos_xy[0] + off[0], pos_xy[1] + off[1], self.z])
            dirs = self.dirs[i] if yaw == 0.0 else self.dirs[i] @ R.T
            mujoco.mj_multiRay(self.m, data, pnt, dirs.ravel(), _GROUP0_ONLY, 1, -1,
                               self._geomid, self._dist, self._normal, self.n, cfg.max_range)
            d = self._dist
            hit = d > 0
            if not hit.any():
                continue
            normals = self._normal.reshape(-1, 3)[hit]
            cos_t = np.abs(np.einsum("ij,ij->i", dirs[hit], normals))
            w = self.beam[i][hit] * self.refl[self._geomid[hit]] * cos_t / d[hit] ** 2
            if w.sum() < self.w_min:
                continue
            mean = float((w * d[hit]).sum() / w.sum())
            if mean > cfg.max_range:
                continue
            readings[i] = mean
            spread[i] = float(np.sqrt((w * (d[hit] - mean) ** 2).sum() / w.sum()))
        return readings, spread

    def corrupt(self, readings: np.ndarray, spread: np.ndarray, noise, rng: np.random.Generator) -> np.ndarray:
        """Sensor noise (proposal §6.5) and bad returns (decision 2). Never empty."""
        cfg = self.cfg
        out = np.asarray(readings, dtype=float).copy()
        if noise.tof_noise_on:
            sigma = np.hypot(noise.tof_a + noise.tof_b * out, noise.tof_spread_gain * spread)
            sigma = np.where(out < cfg.min_range, sigma * noise.tof_close_factor, sigma)
            out += rng.normal(0.0, sigma)
        if noise.bad_return_on:
            bad = rng.random(out.shape) < noise.p_bad_return
            out[bad] += rng.normal(0.0, noise.bad_return_sigma, size=int(bad.sum()))
        return np.clip(out, 0.001, cfg.max_range)
