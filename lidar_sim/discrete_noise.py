"""Discrete Gaussian-derived noise helpers for the grid simulator."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

import config


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Return the Normal(mu, sigma) CDF at x using math.erf."""
    if sigma <= 0.0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def gaussian_three_bin_probs(
    target: float,
    bin_half_width: float,
    sigma: float,
    bias: float = 0.0,
) -> tuple[float, float, float]:
    """Return (under, correct, over) probabilities for a discrete target bin."""
    mu = target + bias
    lower = target - bin_half_width
    upper = target + bin_half_width

    p_under = normal_cdf(lower, mu, sigma)
    p_correct = normal_cdf(upper, mu, sigma) - p_under
    p_over = 1.0 - normal_cdf(upper, mu, sigma)

    return normalize_probs((p_under, p_correct, p_over))


def normalize_probs(probs: Sequence[float]) -> tuple[float, float, float]:
    """Clip and normalize a three-way probability vector defensively."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (3,):
        raise ValueError("expected exactly three probabilities")
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total <= 0.0:
        return 0.0, 1.0, 0.0
    probs /= total
    return tuple(float(p) for p in probs)


def mix_extra_outcomes(
    base_probs: Sequence[float],
    extra_under: float = 0.0,
    extra_over: float = 0.0,
) -> tuple[float, float, float]:
    """Reserve small explicit probabilities for visible rare discrete outcomes."""
    extra_under = max(0.0, float(extra_under))
    extra_over = max(0.0, float(extra_over))
    extra_total = min(extra_under + extra_over, 0.95)
    if extra_total <= 0.0:
        return normalize_probs(base_probs)

    base = np.asarray(normalize_probs(base_probs), dtype=float)
    mixed = base * (1.0 - extra_total)
    mixed[0] += extra_under
    mixed[2] += extra_over
    return normalize_probs(mixed)


def forward_transition_probs(noisy: bool | None = None) -> tuple[float, float, float]:
    """Return (undershoot, correct, overshoot) probabilities for FORWARD."""
    if noisy is None:
        noisy = config.ENABLE_MOTION_NOISE
    if not noisy:
        return 0.0, 1.0, 0.0
    base = gaussian_three_bin_probs(
        config.FORWARD_TARGET_MM,
        config.FORWARD_TRANSITION_BIN_HALF_WIDTH_MM,
        config.FORWARD_SIGMA_MM,
        config.FORWARD_BIAS_MM,
    )
    return mix_extra_outcomes(
        base,
        config.FORWARD_EXTRA_STAY_PROB,
        config.FORWARD_EXTRA_OVERSHOOT_PROB,
    )


def turn_transition_probs(noisy: bool | None = None) -> tuple[float, float, float]:
    """Return (under-turn, correct, over-turn) probabilities for one turn step."""
    if noisy is None:
        noisy = config.ENABLE_MOTION_NOISE
    if not noisy:
        return 0.0, 1.0, 0.0
    base = gaussian_three_bin_probs(
        config.TURN_TARGET_DEG,
        config.TURN_TRANSITION_BIN_HALF_WIDTH_DEG,
        config.TURN_SIGMA_DEG,
        config.TURN_BIAS_DEG,
    )
    return mix_extra_outcomes(
        base,
        config.TURN_EXTRA_UNDERTURN_PROB,
        config.TURN_EXTRA_OVERTURN_PROB,
    )


def sample_three_way(
    probs: Sequence[float],
    rng: np.random.Generator,
) -> int:
    """Sample index 0/1/2 from three probabilities using a numpy Generator."""
    p = np.asarray(probs, dtype=float)
    total = float(p.sum())
    if total <= 0.0:
        p = np.array([0.0, 1.0, 0.0])
    else:
        p = p / total
    return int(rng.choice(3, p=p))
