"""m2 uncertainty sources. Each has an on/off flag and its parameters (m2.3).

The defaults are placeholders, not measurements. m2.4 replaces them with values
from the VL53L4CD datasheet and the predecessor robot's logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    friction_on: bool = False
    mu_lo: float = 0.05            # N, slide-joint dry friction (frictionloss), drawn per run
    mu_hi: float = 0.30
    vel_noise_on: bool = False
    vel_sigma: float = 0.05        # multiplicative: v_true_cmd = v_cmd * (1 + N(0, sigma)), drawn per action
    heading_sigma: float = 0.05    # rad, small rotation of the command; the only source of sideways drift
    stop_jitter_on: bool = False
    stop_jitter_s: float = 0.05    # s, extra coasting at the last command, U[0, stop_jitter_s]
    range_noise_on: bool = False
    range_a: float = 0.002         # m,  sigma(r) = a + b * r, on every reading
    range_b: float = 0.02
    bad_return_on: bool = False
    p_bad_return: float = 0.05     # probability a ray does not return properly (a "miss")
    bad_return_sigma: float = 0.15 # m, the noise of such a reading (about one cell)

    @classmethod
    def all_off(cls) -> "NoiseConfig":
        return cls()

    @classmethod
    def all_on(cls, **overrides) -> "NoiseConfig":
        flags = dict(friction_on=True, vel_noise_on=True, stop_jitter_on=True,
                     range_noise_on=True, bad_return_on=True)
        return cls(**{**flags, **overrides})


def sample_friction(cfg: NoiseConfig, rng: np.random.Generator) -> float:
    return float(rng.uniform(cfg.mu_lo, cfg.mu_hi)) if cfg.friction_on else 0.0


def sample_vel_gain(cfg: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """2x2 command-error matrix G = (1 + eps) * Rot(delta), drawn once per action.
    Per-axis gains alone can never move the robot sideways (the off-axis command is 0),
    so p_drift would be exactly 0; the small rotation delta is what makes drift possible."""
    if not cfg.vel_noise_on:
        return np.eye(2)
    eps = rng.normal(0.0, cfg.vel_sigma)
    dlt = rng.normal(0.0, cfg.heading_sigma)
    c, s = np.cos(dlt), np.sin(dlt)
    return (1.0 + eps) * np.array([[c, -s], [s, c]])


def sample_stop_jitter(cfg: NoiseConfig, rng: np.random.Generator) -> float:
    return float(rng.uniform(0.0, cfg.stop_jitter_s)) if cfg.stop_jitter_on else 0.0


def corrupt_ranges(r: np.ndarray, cfg: NoiseConfig, rng: np.random.Generator,
                   lo: float = 0.001, hi: float = 1.300) -> np.ndarray:
    """Apply sensor noise to noise-free readings (m); the result stays in [lo, hi].

    Team decision (23.11.2026): the real ToF sensors never return "nothing". A miss
    is a ray that does not return properly, so its reading is noisy: with
    probability ``p_bad_return`` the reading gets an extra N(0, bad_return_sigma).
    After discretization a miss is usually a wrong wall distance (or a wall seen
    as far, or a far ray seen as a wall), never an empty reading.
    The m2 draw order is fixed (normal noise for all rays, then the bad-return
    draws), so a seed reproduces the same readings.
    """
    out = np.asarray(r, dtype=float).copy()
    if cfg.range_noise_on:
        out += rng.normal(0.0, cfg.range_a + cfg.range_b * out)
    if cfg.bad_return_on:
        bad = rng.random(out.shape) < cfg.p_bad_return
        out[bad] += rng.normal(0.0, cfg.bad_return_sigma, size=int(bad.sum()))
    return np.clip(out, lo, hi)
