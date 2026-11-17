"""Exact discrete Bayes filter over grid cells (Project B, roadmap §3.4).

The belief is a flat vector ``b`` of length N = rows * cols, indexed
``r * cols + c`` like the motion model's ``T``/``C`` and the sensing model's
``Z``. Heading is not part of the state: sensing needs no rotation.

    predict:  b_bar(s') = sum_s b(s) T[a, s, s'] / (1 - sum_s b(s) C[a, s])
    correct:  b'(s')    ∝ Z[s', o] * b_bar(s')

Prediction is conditioned on "no collision": a run ends on a collision, so a
robot that is still acting knows it did not collide.

Exact zeros are allowed (the discrete model rules cells out). If an observation
or a survival is impossible under the model, which can only happen when the
world does not match the model (e.g. MuJoCo), the filter mixes the belief with
the uniform distribution, ``b <- (1 - eps) b + eps U`` (Burgard et al. 1996),
and the event is counted so runs can report it (``n_fallbacks``).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

import config
from motion_model import Action, cell_index

log = logging.getLogger(__name__)


def uniform_belief(rows: int, cols: int, exclude: Iterable[tuple[int, int]] = ()) -> np.ndarray:
    """Uniform belief over all cells except ``exclude`` (e.g. the goal, D10)."""
    b = np.ones(rows * cols, dtype=np.float64)
    for r, c in exclude:
        b[cell_index(r, c, cols)] = 0.0
    total = b.sum()
    if total == 0.0:
        raise ValueError("every cell is excluded")
    return b / total


def _mix_uniform(b: np.ndarray, eps: float) -> np.ndarray:
    return (1.0 - eps) * b + eps / b.size


def predict(
    b: np.ndarray, T: np.ndarray, C: np.ndarray, action: Action, eps: float | None = None
) -> tuple[np.ndarray, bool]:
    """Motion update conditioned on no collision.

    Returns ``(b_bar, ok)``. ``ok`` is False when survival had probability 0
    under the model. ``b_bar`` is then the uniform-mixed prediction of the
    uniform-mixed prior (no cell is ruled out).
    """
    action = Action(action)
    if not action.is_move:
        raise ValueError("DECLARE ends the run; there is nothing to predict")
    eps = config.BELIEF_FALLBACK_EPS if eps is None else eps
    b_bar = b @ T[action]
    total = b_bar.sum()          # = 1 - sum_s b(s) C[a, s]
    if total > 0.0:
        return b_bar / total, True
    b_bar = _mix_uniform(b, eps) @ T[action]
    total = b_bar.sum()
    if total > 0.0:
        return _mix_uniform(b_bar / total, eps), False
    return np.full_like(b, 1.0 / b.size), False


def correct(
    b_bar: np.ndarray, Z: np.ndarray, o: int, eps: float | None = None
) -> tuple[np.ndarray, bool]:
    """Observation update ``b' ∝ Z[:, o] * b_bar``.

    Returns ``(b', ok)``. ``ok`` is False when ``o`` had probability 0 under
    ``b_bar``. The filter then mixes ``b_bar`` with the uniform distribution
    before correcting, and falls back to uniform if ``o`` is impossible in
    every cell.
    """
    eps = config.BELIEF_FALLBACK_EPS if eps is None else eps
    likelihood = Z[:, int(o)]
    post = likelihood * b_bar
    total = post.sum()
    if total > 0.0:
        return post / total, True
    post = likelihood * _mix_uniform(b_bar, eps)
    total = post.sum()
    if total > 0.0:
        return post / total, False
    return np.full_like(b_bar, 1.0 / b_bar.size), False


def observation_probabilities(b_bar: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``P(o | b_bar) = sum_s b_bar(s) Z[s, o]`` for every o, shape ``(|O|,)``."""
    return b_bar @ Z


class BayesFilter:
    """Holds the model matrices, the current belief and the fallback counter.

    Usage:
        f = BayesFilter(T, C, Z)
        f.reset(b0, first_obs)          # correct with the observation at reset
        f.update(action, obs)           # after every move that did not collide
    """

    def __init__(self, T: np.ndarray, C: np.ndarray, Z: np.ndarray, eps: float | None = None):
        n = Z.shape[0]
        if T.shape[1:] != (n, n) or C.shape[1] != n:
            raise ValueError(f"T {T.shape}, C {C.shape} and Z {Z.shape} disagree on N")
        self.T, self.C, self.Z = T, C, Z
        self.eps = config.BELIEF_FALLBACK_EPS if eps is None else eps
        self.belief: np.ndarray | None = None
        self.n_fallbacks = 0

    def reset(self, b0: np.ndarray, obs: int | None = None) -> np.ndarray:
        b0 = np.asarray(b0, dtype=np.float64)
        if b0.shape != (self.Z.shape[0],) or not np.isclose(b0.sum(), 1.0) or (b0 < 0).any():
            raise ValueError("b0 must be a probability vector of length N")
        self.belief = b0.copy()
        self.n_fallbacks = 0
        if obs is not None:
            self.belief = self._correct(self.belief, obs)
        return self.belief

    def update(self, action: Action, obs: int | None) -> np.ndarray:
        """Predict with ``action``, then correct with ``obs`` (skipped if None)."""
        if self.belief is None:
            raise RuntimeError("call reset() before update()")
        b_bar, ok = predict(self.belief, self.T, self.C, action, self.eps)
        if not ok:
            self._fallback("survival after %s", Action(action).name)
        self.belief = b_bar if obs is None else self._correct(b_bar, obs)
        return self.belief

    def _correct(self, b_bar: np.ndarray, obs: int) -> np.ndarray:
        b, ok = correct(b_bar, self.Z, obs, self.eps)
        if not ok:
            self._fallback("observation %d", obs)
        return b

    def _fallback(self, what: str, arg) -> None:
        self.n_fallbacks += 1
        log.warning("impossible " + what + " under the model; mixing with uniform", arg)
