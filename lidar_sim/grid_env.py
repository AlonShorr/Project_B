"""Headless grid environment for Project B episodes (motion only).

An episode starts from a uniformly random non-goal cell (kidnapped robot) or
a given cell. Each ``step`` either moves the robot with the noisy motion model
or handles the terminal DECLARE action:

- DECLARE in a goal cell             -> SUCCESS
- DECLARE elsewhere                  -> WRONG_DECLARATION
- a move that crosses a closed edge  -> COLLISION (wall_mode="collide")
- ``max_steps`` actions without end  -> TIMEOUT

Every terminal outcome is a separate type in ``Outcome`` and is stored in the
episode log, so batch runs can count collisions apart from wrong declarations.

Observations (t4) are opt-in: pass ``sensing=SensingParams(...)`` and ``obs`` is
an observation index from ``sensing.py``, received at reset and after every
move that does not collide. DECLARE and terminal steps give ``obs = None``
(roadmap D7). Without ``sensing``, ``obs`` is always None.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

import config
from motion_model import (
    Action,
    MotionParams,
    MoveResult,
    Outcome,
    cell_index,
    sample_move,
)
from sensing import SensingParams, ray_walls, sample_observation

Cell = tuple[int, int]


@dataclass
class EpisodeLog:
    """Record of one episode, filled in as the episode runs."""

    seed: int | None
    start: Cell
    outcome: Outcome = Outcome.RUNNING
    steps: int = 0
    trajectory: list[Cell] = field(default_factory=list)
    moves: list[MoveResult] = field(default_factory=list)
    final_cell: Cell | None = None
    collision: MoveResult | None = None   # the move that collided, if any
    observations: list[int] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.outcome.is_failure

    def as_row(self) -> dict[str, Any]:
        """Flat dict for CSV output."""
        return {
            "seed": self.seed,
            "start_r": self.start[0],
            "start_c": self.start[1],
            "outcome": self.outcome.value,
            "failed": self.failed,
            "steps": self.steps,
            "final_r": None if self.final_cell is None else self.final_cell[0],
            "final_c": None if self.final_cell is None else self.final_cell[1],
            "collision_action": None if self.collision is None else self.collision.action.name,
            "collision_direction": None if self.collision is None else self.collision.direction.name,
            "collision_k": None if self.collision is None else self.collision.k,
        }


@dataclass(frozen=True)
class StepResult:
    obs: int | None          # observation index, or None (no sensing / DECLARE / terminal)
    outcome: Outcome
    info: dict[str, Any]


class GridEnv:
    """Samples episodes from the motion model on one wall_map."""

    def __init__(
        self,
        wall_map: np.ndarray,
        goal_cells: Iterable[Cell],
        params: MotionParams | None = None,
        max_steps: int | None = None,
        sensing: SensingParams | None = None,
    ):
        self.wall_map = np.asarray(wall_map)
        self.rows, self.cols = self.wall_map.shape[:2]
        self.goal_cells = frozenset((int(r), int(c)) for r, c in goal_cells)
        if not self.goal_cells:
            raise ValueError("goal_cells must not be empty")
        for r, c in self.goal_cells:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"goal cell {(r, c)} is outside the map")
        self.params = params if params is not None else MotionParams.from_config()
        self.max_steps = config.MAX_EPISODE_STEPS if max_steps is None else int(max_steps)
        self.sensing = sensing
        self._ray_table = None if sensing is None else ray_walls(self.wall_map, sensing.R)

        self.rng: np.random.Generator | None = None
        self.cell: Cell | None = None
        self.log: EpisodeLog | None = None

    # -- episode control ---------------------------------------------------

    def start_cells(self) -> list[Cell]:
        """Possible kidnapped start cells: every non-goal cell."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.goal_cells
        ]

    def reset(self, seed: int | None = None, start: Cell | None = None) -> StepResult:
        """Start a new episode. All randomness comes from a Generator seeded here."""
        self.rng = np.random.default_rng(seed)
        if start is None:
            candidates = self.start_cells()
            start = candidates[int(self.rng.integers(len(candidates)))]
        start = (int(start[0]), int(start[1]))
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"start cell {start} is outside the map")
        self.cell = start
        self.log = EpisodeLog(seed=seed, start=start, trajectory=[start])
        return StepResult(obs=self._observe(), outcome=Outcome.RUNNING, info={"cell": start})

    @property
    def done(self) -> bool:
        return self.log is not None and self.log.outcome.is_terminal

    def step(self, action: Action) -> StepResult:
        if self.log is None or self.cell is None or self.rng is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError(f"episode already ended with {self.log.outcome.name}")

        action = Action(action)
        self.log.steps += 1
        info: dict[str, Any] = {"action": action}

        if action is Action.DECLARE:
            outcome = Outcome.SUCCESS if self.cell in self.goal_cells else Outcome.WRONG_DECLARATION
            return self._finish(outcome, info)

        move = sample_move(self.wall_map, *self.cell, action, self.params, self.rng)
        self.cell = (move.r, move.c)
        self.log.moves.append(move)
        self.log.trajectory.append(self.cell)
        info["move"] = move

        if move.collided:
            self.log.collision = move
            return self._finish(Outcome.COLLISION, info)
        if self.log.steps >= self.max_steps:
            return self._finish(Outcome.TIMEOUT, info)

        info["cell"] = self.cell
        return StepResult(obs=self._observe(), outcome=Outcome.RUNNING, info=info)

    def _observe(self) -> int | None:
        """Sample an observation at the current cell (None without sensing)."""
        if self.sensing is None:
            return None
        s = cell_index(*self.cell, self.cols)
        obs = sample_observation(self._ray_table, s, self.sensing, self.rng)
        self.log.observations.append(obs)
        return obs

    def _finish(self, outcome: Outcome, info: dict[str, Any]) -> StepResult:
        self.log.outcome = outcome
        self.log.final_cell = self.cell
        info["cell"] = self.cell
        info["log"] = self.log
        return StepResult(obs=None, outcome=outcome, info=info)


def outcome_counts(logs: Iterable[EpisodeLog]) -> dict[str, int]:
    """Count episodes per outcome, including zero counts for every outcome type."""
    counts = Counter(log.outcome for log in logs)
    return {o.value: counts.get(o, 0) for o in Outcome if o is not Outcome.RUNNING}
