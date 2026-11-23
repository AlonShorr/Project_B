"""Project A heuristic ported to the Project B model (roadmap t7, ROADMAP_TECHNICAL.md §5.1).

Project A localized a lost robot with greedy one-step information gain, then
drove to the goal with A*. The port keeps that two-mode structure on the
Project B POMDP model (cells only, 8 absolute moves, range-R sensing, collisions
that end the run, an explicit DECLARE):

    every step:
        b <- Bayes update with the last move and the new observation
        if b(G) >= tau_declare:                       DECLARE
        LOCALIZING -> TRAVELING  when localized(b)    (t1.2 switching rule)
        TRAVELING  -> LOCALIZING when P_max < tau_lost (hysteresis)
        safe(a)  <=>  sum_s b(s) C[a, s] <= eps_col   (new vs Project A)
        LOCALIZING: argmax over safe moves of  H(b) - E_o[H(b^{a,o})] - move_cost
        TRAVELING:  first step of a BFS shortest path from the MAP cell to G

    localized(b) <=> P_max >= tau_loc and P_far / P_max < rho_amb, where P_far
    is the most likely cell at Chebyshev distance >= sep_min from the arg-max
    (Project A's ``localization.dominant_position_modes``).

What changed against Project A (see the table in ROADMAP_TECHNICAL.md §5.1):
- The expected posterior entropy is exact: O is finite, so
  ``E_o[H] = sum_o P(o) H(b^o)`` is computed for every observation at once,
  instead of Project A's noiseless top-16-hypotheses approximation.
- All 8 moves cost the same, so ``move_cost`` cannot rank moves against each
  other as it did against Project A's cheap rotations. It is used as the price
  of information instead: when no safe move gains more than ``move_cost`` nats,
  a localizing robot makes progress toward the goal from its MAP cell.
- Collision risk is gated by ``eps_col``. When every move is riskier than
  ``eps_col`` the least risky move is taken and counted (``n_forced``).
- DECLARE is a belief decision (``tau_declare``), not "the true pose is the goal".

This module has no dependency on the runner; ``policies.ProjectAPolicy`` wraps
``ProjectAAgent`` for it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from belief import BayesFilter, uniform_belief
from motion_model import (
    DIR_DELTAS,
    MOVE_ACTIONS,
    Action,
    MotionParams,
    build_transition_model,
    can_step,
    cell_index,
)
from sensing import SensingParams, build_observation_model

Cell = tuple[int, int]


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectAParams:
    """Thresholds of the heuristic (the ``projecta`` section of the config).

    tau_loc, rho_amb, sep_min: LOCALIZING -> TRAVELING switch (t1.2)
    tau_lost:    TRAVELING -> LOCALIZING when P_max drops below it
    tau_declare: DECLARE when the belief mass on the goal cells reaches it
    eps_col:     largest believed collision probability allowed for one move
    move_cost:   price of a move in nats while localizing
    """

    tau_loc: float
    rho_amb: float
    tau_lost: float
    tau_declare: float
    eps_col: float
    move_cost: float
    sep_min: int

    def __post_init__(self):
        for name in ("tau_loc", "rho_amb", "tau_lost", "tau_declare", "eps_col"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
            object.__setattr__(self, name, value)
        if self.move_cost < 0.0:
            raise ValueError(f"move_cost must be >= 0, got {self.move_cost}")
        object.__setattr__(self, "move_cost", float(self.move_cost))
        if int(self.sep_min) != self.sep_min or self.sep_min < 1:
            raise ValueError(f"sep_min must be an integer >= 1, got {self.sep_min}")
        object.__setattr__(self, "sep_min", int(self.sep_min))
        if self.tau_lost >= self.tau_loc:
            raise ValueError(
                f"tau_lost ({self.tau_lost}) must be below tau_loc ({self.tau_loc}) "
                "so the switch has hysteresis"
            )

    @classmethod
    def from_mapping(cls, values: Any) -> "ProjectAParams":
        """From a dict or any object with the same attributes (e.g. sim_config.ProjectAConfig)."""
        if isinstance(values, cls):
            return values
        names = [f.name for f in fields(cls)]
        if isinstance(values, Mapping):
            missing = [n for n in names if n not in values]
            unknown = sorted(set(values) - set(names))
            if missing or unknown:
                raise ValueError(f"projecta params: missing {missing}, unknown {unknown}")
            return cls(**{n: values[n] for n in names})
        return cls(**{n: getattr(values, n) for n in names})

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------
# Information measures (vectorized, exact zeros allowed)
# --------------------------------------------------------------------------

def _xlogx(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    pos = x > 0.0
    out[pos] = x[pos] * np.log(x[pos])
    return out


def belief_entropy(b: np.ndarray) -> float:
    """Shannon entropy in nats; ``ln k`` for k equally likely cells."""
    b = np.asarray(b, dtype=np.float64)
    total = b.sum()
    if total <= 0.0:
        raise ValueError("belief has no mass")
    return float(-_xlogx(b / total).sum())


def expected_posterior_entropy(b_bar: np.ndarray, Z: np.ndarray) -> float:
    """``E_o[H(b^o)] = sum_o P(o) H(b^o)`` with ``b^o ∝ Z[:, o] * b_bar``.

    With ``M[s, o] = b_bar(s) Z[s, o]`` and ``P(o) = sum_s M[s, o]``:
        sum_o P(o) H(b^o) = -sum_{s,o} M log M + sum_o P(o) log P(o)
    """
    M = b_bar[:, None] * Z
    p_o = M.sum(axis=0)
    return float(-_xlogx(M).sum() + _xlogx(p_o).sum())


@dataclass(frozen=True)
class MoveScore:
    action: Action
    risk: float     # believed collision probability, sum_s b(s) C[a, s]
    gain: float     # H(b) - E_o[H(b^{a,o})], given no collision (nan if the move surely collides)


def score_moves(
    b: np.ndarray, T: np.ndarray, C: np.ndarray, Z: np.ndarray, moves: Sequence[Action]
) -> list[MoveScore]:
    """Collision risk and expected information gain of every move in ``moves``."""
    h = belief_entropy(b)
    scores = []
    for a in moves:
        risk = float(b @ C[a])
        b_bar = b @ T[a]
        survive = b_bar.sum()
        if survive <= 0.0:
            scores.append(MoveScore(a, risk, float("nan")))
            continue
        gain = h - expected_posterior_entropy(b_bar / survive, Z)
        scores.append(MoveScore(a, risk, gain))
    return scores


# --------------------------------------------------------------------------
# Modes and the switching rule
# --------------------------------------------------------------------------

def dominant_modes(b: np.ndarray, rows: int, cols: int, sep_min: int = 2) -> tuple[int, float, float]:
    """Return ``(peak index, P_max, P_far)``.

    ``P_far`` is the largest probability among cells at Chebyshev distance
    >= ``sep_min`` from the peak (0 if there are none). Port of Project A's
    ``localization.dominant_position_modes`` to a flat belief over cells.
    """
    peak = int(np.argmax(b))
    pr, pc = divmod(peak, cols)
    grid = b.reshape(rows, cols)
    rr = np.abs(np.arange(rows) - pr)[:, None]
    cc = np.abs(np.arange(cols) - pc)[None, :]
    far = np.maximum(rr, cc) >= sep_min
    p_far = float(grid[far].max()) if far.any() else 0.0
    return peak, float(b[peak]), p_far


def is_localized(b: np.ndarray, rows: int, cols: int, p: ProjectAParams) -> bool:
    """The t1.2 rule: a clear peak and no comparable distant rival."""
    _, p_max, p_far = dominant_modes(b, rows, cols, p.sep_min)
    return p_max >= p.tau_loc and p_far < p.rho_amb * p_max


# --------------------------------------------------------------------------
# Travel
# --------------------------------------------------------------------------

def goal_distances(wall_map: np.ndarray, goal_cells: Iterable[Cell], moves: Sequence[Action]) -> np.ndarray:
    """BFS step count from every cell to the nearest goal (inf if unreachable)."""
    from collections import deque

    rows, cols = wall_map.shape[:2]
    dist = np.full((rows, cols), np.inf)
    queue = deque()
    for r, c in goal_cells:
        dist[r, c] = 0.0
        queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for a in moves:
            if can_step(wall_map, r, c, a):   # symmetric, so outward BFS = distance to goal
                dr, dc = DIR_DELTAS[int(a)]
                if dist[r + dr, c + dc] == np.inf:
                    dist[r + dr, c + dc] = dist[r, c] + 1.0
                    queue.append((r + dr, c + dc))
    return dist


def next_cell_distances(wall_map: np.ndarray, dist: np.ndarray, cell: Cell, moves: Sequence[Action]) -> dict[Action, float]:
    """Distance to the goal after one nominal step from ``cell`` (blocked moves left out)."""
    r, c = cell
    out = {}
    for a in moves:
        if can_step(wall_map, r, c, a):
            dr, dc = DIR_DELTAS[int(a)]
            out[a] = float(dist[r + dr, c + dc])
    return out


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

class Mode(Enum):
    LOCALIZING = "localizing"
    TRAVELING = "traveling"


@dataclass
class Model:
    """Everything the agent precomputes per maze and noise setting."""

    wall_map: np.ndarray
    goal_cells: frozenset[Cell]
    rows: int
    cols: int
    T: np.ndarray
    C: np.ndarray
    Z: np.ndarray            # only the observation columns that can occur
    goal_mask: np.ndarray    # (N,) bool
    dist: np.ndarray         # (rows, cols) BFS steps to the goal set
    moves: tuple[Action, ...]

    @classmethod
    def build(
        cls,
        wall_map: np.ndarray,
        goal_cells: Iterable[Cell],
        motion: MotionParams,
        sensing: SensingParams,
        moves: Sequence[Action] = MOVE_ACTIONS,
    ) -> "Model":
        wall_map = np.asarray(wall_map)
        rows, cols = wall_map.shape[:2]
        goal_cells = frozenset((int(r), int(c)) for r, c in goal_cells)
        T, C = build_transition_model(wall_map, motion)
        Z = build_observation_model(wall_map, sensing)
        goal_mask = np.zeros(rows * cols, dtype=bool)
        for r, c in goal_cells:
            goal_mask[cell_index(r, c, cols)] = True
        return cls(
            wall_map=wall_map,
            goal_cells=goal_cells,
            rows=rows,
            cols=cols,
            T=T,
            C=C,
            Z=Z,
            goal_mask=goal_mask,
            dist=goal_distances(wall_map, goal_cells, moves),
            moves=tuple(moves),
        )

    @property
    def Z_used(self) -> np.ndarray:
        """Z without the observation columns that never occur (same expectations, faster)."""
        if not hasattr(self, "_Z_used"):
            self._Z_used = self.Z[:, self.Z.any(axis=0)]
        return self._Z_used


class ProjectAAgent:
    """The §5.1 state machine over an exact Bayes filter.

    Usage (the runner's order):
        agent.reset()                  # start of an episode
        a = agent.act(obs)             # obs from reset, then after every move
    """

    def __init__(self, model: Model, params: ProjectAParams):
        self.model = model
        self.p = params
        self.filter = BayesFilter(model.T, model.C, model.Z)
        self.reset()

    def reset(self) -> None:
        m = self.model
        # D10: the kidnapped robot is never started in a goal cell.
        self.b0 = uniform_belief(m.rows, m.cols, exclude=m.goal_cells)
        self.filter.reset(self.b0)
        self.mode = Mode.LOCALIZING
        self.last_action: Action | None = None
        self.t = 0
        self.steps_to_loc: int | None = None
        self.n_forced = 0         # moves taken although every move was riskier than eps_col
        self.n_mode_switches = 0
        self.trace: list[tuple[Mode, Action]] = []

    @property
    def belief(self) -> np.ndarray:
        return self.filter.belief

    @property
    def p_goal(self) -> float:
        return float(self.belief[self.model.goal_mask].sum())

    @property
    def n_fallbacks(self) -> int:
        return self.filter.n_fallbacks

    # -- one decision --------------------------------------------------------

    def act(self, obs: int | None) -> Action:
        if self.last_action is None:
            if obs is not None:
                self.filter.reset(self.b0, obs)
        else:
            self.filter.update(self.last_action, obs)

        action = self._decide()
        self.trace.append((self.mode, action))
        self.last_action = action if action.is_move else None
        self.t += 1
        return action

    def _decide(self) -> Action:
        m, p, b = self.model, self.p, self.belief
        if self.p_goal >= p.tau_declare:
            return Action.DECLARE

        if self.mode is Mode.LOCALIZING and is_localized(b, m.rows, m.cols, p):
            self._switch(Mode.TRAVELING)
        elif self.mode is Mode.TRAVELING and float(b.max()) < p.tau_lost:
            self._switch(Mode.LOCALIZING)

        scores = score_moves(b, m.T, m.C, m.Z_used, m.moves)
        safe = [s for s in scores if s.risk <= p.eps_col and not np.isnan(s.gain)]
        if not safe:
            self.n_forced += 1
            return min(scores, key=lambda s: (s.risk, -np.nan_to_num(s.gain, nan=-np.inf))).action

        if self.mode is Mode.TRAVELING:
            a = self._travel(safe)
            return a if a is not None else self._localize(safe)
        return self._localize(safe, allow_travel=True)

    def _switch(self, mode: Mode) -> None:
        self.mode = mode
        self.n_mode_switches += 1
        if mode is Mode.TRAVELING and self.steps_to_loc is None:
            self.steps_to_loc = self.t

    def _localize(self, safe: list[MoveScore], allow_travel: bool = False) -> Action:
        """Greedy information gain; travel instead when no move is worth ``move_cost``."""
        best = max(safe, key=lambda s: s.gain)   # ties: first in N, NE, E, ... order
        if allow_travel and best.gain - self.p.move_cost <= 0.0:
            a = self._travel(safe)
            if a is not None:
                return a
        return best.action

    def _travel(self, safe: list[MoveScore]) -> Action | None:
        """A safe move that brings the MAP cell closer to the goal (or keeps the distance).

        Returns None when the MAP cell is a goal cell or no safe move helps.
        """
        m = self.model
        cell = divmod(int(np.argmax(self.belief)), m.cols)
        here = float(m.dist[cell])
        if here == 0.0:
            return None
        after = next_cell_distances(m.wall_map, m.dist, cell, m.moves)
        by_action = {s.action: s for s in safe}
        options = [(d, by_action[a].risk, int(a)) for a, d in after.items() if a in by_action and d <= here]
        if not options:
            return None
        d, _, a = min(options)
        if d == here:
            # Only sideways moves are safe; prefer one that also gains information.
            sideways = [by_action[Action(a)] for dd, _, a in options if dd == here]
            return max(sideways, key=lambda s: s.gain).action
        return Action(a)
