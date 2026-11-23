"""GridEnv's interface on MuJoCo physics (m1.3: one interface, two simulators).

``MujocoGridEnv`` has the same constructor arguments, attributes and methods
that ``runner.EpisodeRunner``, ``policies.EnvInfo.from_env`` and the policies
use on ``grid_env.GridEnv``, so the same policy code runs on both unchanged:

    reset(seed, start=None) -> StepResult(obs, outcome, info)
    step(action)            -> StepResult(obs, outcome, info)
    done, log (grid_env.EpisodeLog), start_cells()
    wall_map, rows, cols, goal_cells, params, max_steps, sensing

The difference is where outcomes come from. ``params`` and ``sensing`` are the
models the POLICY assumes (from the experiment config); the robot's actual
motion and readings come from MuJoCo and the m2 noise in ``noise``. Every move
action is a one-cell command; 0, 1 or 2 cells and drift emerge from physics.

Randomness: ``reset(seed)`` draws the start cell exactly as GridEnv does (same
candidate order, ``default_rng(seed)``), so both simulators start each seed in
the same cell. The same generator then drives the m2 noise, and the MuJoCo
state is reset, so an episode depends on its seed only.
"""

from __future__ import annotations

from typing import Any, Iterable

import mujoco
import numpy as np

from .noise import NoiseConfig
from .noisy_env import NoisyMazeSim
import config
from grid_env import EpisodeLog, StepResult
from motion_model import DIR_DELTAS, Action, MotionParams, MoveResult, Outcome
from sensing import SensingParams, encode_observation

Cell = tuple[int, int]
_DELTA_TO_ACTION = {d: Action(i) for i, d in enumerate(DIR_DELTAS)}


class MujocoGridEnv:
    """Episodes on one wall_map, simulated in MuJoCo."""

    def __init__(
        self,
        wall_map: np.ndarray,
        goal_cells: Iterable[Cell],
        params: MotionParams | None = None,
        max_steps: int | None = None,
        sensing: SensingParams | None = None,
        noise: NoiseConfig | None = None,
        controller: dict[str, Any] | None = None,
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
        R = sensing.R if sensing is not None else config.SENSING_RANGE_R
        self.sim = NoisyMazeSim(self.wall_map, (0, 0), R=R, noise=noise, seed=0, **(controller or {}))
        self.rng: np.random.Generator | None = None
        self.cell: Cell | None = None
        self.log: EpisodeLog | None = None

    # -- episode control (same as GridEnv) -----------------------------------
    def start_cells(self) -> list[Cell]:
        """Possible kidnapped start cells: every non-goal cell (GridEnv order)."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.goal_cells
        ]

    def reset(self, seed: int | None = None, start: Cell | None = None) -> StepResult:
        self.rng = np.random.default_rng(seed)
        if start is None:
            candidates = self.start_cells()
            start = candidates[int(self.rng.integers(len(candidates)))]
        start = (int(start[0]), int(start[1]))
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"start cell {start} is outside the map")
        sim = self.sim
        mujoco.mj_resetData(sim.model, sim.data)   # no state (warm start, time) carries over
        sim.rng = self.rng                         # m2 noise from the episode's generator
        sim.new_run()                              # per-episode friction draw
        sim.teleport(start)
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

        res = self.sim.step_action(action, 1)
        move = self._move_result(action, self.cell, res.cell, res.collided)
        self.cell = res.cell
        self.log.moves.append(move)
        self.log.trajectory.append(self.cell)
        info["move"] = move
        info["sim_time"] = res.sim_time
        if res.collided:
            self.log.collision = move
            return self._finish(Outcome.COLLISION, info)
        if self.log.steps >= self.max_steps:
            return self._finish(Outcome.TIMEOUT, info)
        info["cell"] = self.cell
        return StepResult(obs=self._observe(), outcome=Outcome.RUNNING, info=info)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _move_result(action: Action, start: Cell, end: Cell, collided: bool) -> MoveResult:
        """Describe a physical move in motion_model terms: the realized direction and
        step count k are read off the displacement (k = 0 keeps the commanded direction)."""
        dr, dc = end[0] - start[0], end[1] - start[1]
        k = max(abs(dr), abs(dc))
        direction = action if k == 0 else _DELTA_TO_ACTION.get((int(np.sign(dr)), int(np.sign(dc))), action)
        return MoveResult(action=action, direction=direction, k=k, start=start,
                          r=end[0], c=end[1], cells_moved=k, collided=collided)

    def _observe(self) -> int | None:
        if self.sensing is None:
            return None
        obs = encode_observation(tuple(int(v) for v in self.sim.observe()), self.sensing)
        self.log.observations.append(obs)
        return obs

    def _finish(self, outcome: Outcome, info: dict[str, Any]) -> StepResult:
        self.log.outcome = outcome
        self.log.final_cell = self.cell
        info["cell"] = self.cell
        info["log"] = self.log
        return StepResult(obs=None, outcome=outcome, info=info)


def make_mujoco_env(cfg, maze, mj_cfg: dict[str, Any]) -> MujocoGridEnv:
    """``runner`` env factory: ``functools.partial(make_mujoco_env, mj_cfg=...)``.
    The experiment config gives the maze, max_steps and the models the policy assumes;
    ``mj_cfg`` (from mj_config.load_mj_config) gives the controller and the noise."""
    from .mj_config import noise_config

    return MujocoGridEnv(
        maze.wall_map,
        maze.goal_cells,
        params=cfg.to_motion_params(),
        max_steps=cfg.episode.max_steps,
        sensing=cfg.to_sensing_params(),
        noise=noise_config(mj_cfg),
        controller=dict(mj_cfg["controller"]),
    )
