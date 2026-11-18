"""Policy interface for headless episodes, plus two placeholder policies (t5).

A policy is created once per worker and reused across episodes:

    policy = make_policy("random", {"declare_after": 20})
    policy.reset(env_info, rng)          # start of every episode
    action = policy.act(obs, info)       # every step until the episode ends

``env_info`` is what the robot may know in advance: the map, the goal cells
and the noise model. ``rng`` is the policy's own generator, separate from the
environment's, so changing the policy never changes the start cells.

``info`` is filtered by the runner: only policies with ``privileged = True``
(oracles) see the true cell. Everything else gets the step index and the last
action. Observations (``obs``) come from the sensing model (t4); until then
they are None.

``episode_stats()`` returns optional per-episode metrics for the CSV
(``steps_to_loc``, ``final_P_goal``, ``n_fallbacks``); placeholders return {}.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np

from motion_model import DIR_DELTAS, MOVE_ACTIONS, Action, MotionParams, can_step
from sim_config import check_bool, check_int

Cell = tuple[int, int]
CARDINAL_ACTIONS: tuple[Action, ...] = (Action.N, Action.E, Action.S, Action.W)


@dataclass(frozen=True)
class EnvInfo:
    """Static knowledge available to a policy at the start of an episode."""

    wall_map: np.ndarray
    goal_cells: frozenset[Cell]
    rows: int
    cols: int
    motion: MotionParams
    max_steps: int

    @classmethod
    def from_env(cls, env) -> "EnvInfo":
        return cls(
            wall_map=env.wall_map,
            goal_cells=env.goal_cells,
            rows=env.rows,
            cols=env.cols,
            motion=env.params,
            max_steps=env.max_steps,
        )


class Policy:
    """Base class. Subclasses set ``name`` and ``DEFAULTS`` and implement ``act``.

    ``DEFAULTS`` names the parameters and fills them in only when a policy is
    built directly in code. Experiments take every value from the config file
    (``policy.params.<name>`` in configs/default.yaml).
    """

    name: ClassVar[str] = ""
    DEFAULTS: ClassVar[dict[str, Any]] = {}
    privileged: ClassVar[bool] = False   # True: ``info`` includes the true cell

    def __init__(self, **params: Any):
        self.params = resolve_params(self.name, params, cls=type(self))
        self.env_info: EnvInfo | None = None
        self.rng: np.random.Generator | None = None

    def reset(self, env_info: EnvInfo, rng: np.random.Generator) -> None:
        self.env_info = env_info
        self.rng = rng

    def act(self, obs: Any, info: Mapping[str, Any]) -> Action:
        raise NotImplementedError

    def episode_stats(self) -> dict[str, Any]:
        return {}

    @classmethod
    def check_params(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and type-convert params (defaults already filled in)."""
        return dict(params)


class RandomPolicy(Policy):
    """Uniformly random moves, then DECLARE after ``declare_after`` moves."""

    name = "random"
    DEFAULTS = {"declare_after": 20, "diagonal": True}

    @classmethod
    def check_params(cls, params):
        return {
            "declare_after": check_int(params["declare_after"], "declare_after", lo=0),
            "diagonal": check_bool(params["diagonal"], "diagonal"),
        }

    def reset(self, env_info, rng):
        super().reset(env_info, rng)
        self._moves = MOVE_ACTIONS if self.params["diagonal"] else CARDINAL_ACTIONS
        self._t = 0

    def act(self, obs, info):
        if self._t >= self.params["declare_after"]:
            return Action.DECLARE
        self._t += 1
        return self._moves[int(self.rng.integers(len(self._moves)))]


class OracleBFSPolicy(Policy):
    """Privileged reference baseline: knows its true cell, ignores motion noise.

    Distances to the goal set are a multi-source BFS over wall-respecting single
    steps (``can_step``, so diagonals follow the corner rule). Each step the
    policy takes the legal move that most reduces the distance (ties: first in
    N, NE, E, ... order), and declares once it is in a goal cell, so it never
    makes a wrong declaration.

    It is NOT an upper bound on performance. It plans as if every move went
    exactly one cell in the commanded direction, so drift and k = 2 still make
    it collide now and then (about 0.03% of episodes on medium/m10_a with the
    t1.3 defaults; about 31% with the old p_drift = 0.1). A noise-aware policy
    can do better on collisions; the exact floor is in defaults_analysis.py.
    """

    name = "oracle_bfs"
    DEFAULTS = {"diagonal": True}
    privileged = True

    @classmethod
    def check_params(cls, params):
        return {"diagonal": check_bool(params["diagonal"], "diagonal")}

    def __init__(self, **params):
        super().__init__(**params)
        self._cache_key: tuple | None = None
        self._dist: np.ndarray | None = None

    def reset(self, env_info, rng):
        super().reset(env_info, rng)
        self._moves = MOVE_ACTIONS if self.params["diagonal"] else CARDINAL_ACTIONS
        wall_map = np.ascontiguousarray(env_info.wall_map)
        key = (wall_map.shape, wall_map.tobytes(), env_info.goal_cells, self._moves)
        if key != self._cache_key:
            self._dist = bfs_distances(env_info.wall_map, env_info.goal_cells, self._moves)
            self._cache_key = key

    def act(self, obs, info):
        cell = info["cell"]
        if cell in self.env_info.goal_cells:
            return Action.DECLARE
        r, c = cell
        best, best_d = None, self._dist[r, c]
        for a in self._moves:
            if not can_step(self.env_info.wall_map, r, c, a):
                continue
            dr, dc = _delta(a)
            d = self._dist[r + dr, c + dc]
            if d < best_d:
                best, best_d = a, d
        if best is None:   # goal unreachable from here: give up
            return Action.DECLARE
        return best


def _delta(a: Action) -> tuple[int, int]:
    return DIR_DELTAS[int(a)]


def bfs_distances(wall_map: np.ndarray, goal_cells, moves=MOVE_ACTIONS) -> np.ndarray:
    """Number of single steps from every cell to the nearest goal (inf if unreachable).

    Wall checks are symmetric (an edge or a corner is open from both sides), so
    a BFS outward from the goals gives the distance *to* the goals.
    """
    rows, cols = wall_map.shape[:2]
    dist = np.full((rows, cols), np.inf)
    queue = deque()
    for r, c in goal_cells:
        dist[r, c] = 0
        queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for a in moves:
            if can_step(wall_map, r, c, a):
                dr, dc = _delta(a)
                nr, nc = r + dr, c + dc
                if dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
    return dist


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

POLICIES: dict[str, type[Policy]] = {
    RandomPolicy.name: RandomPolicy,
    OracleBFSPolicy.name: OracleBFSPolicy,
}
POLICY_NAMES: tuple[str, ...] = tuple(POLICIES)


def resolve_params(
    name: str,
    params: Mapping[str, Any],
    cls: type[Policy] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate parameters. Unknown names raise ValueError.

    ``strict=True`` (used for configs) also requires every parameter, so the
    values come from the YAML file only. Otherwise missing ones take the class
    ``DEFAULTS`` (convenient when a policy is built directly in code).
    """
    cls = cls or POLICIES.get(name)
    if cls is None:
        raise ValueError(f"unknown policy {name!r}; known: {', '.join(POLICY_NAMES)}")
    if not isinstance(params, Mapping):
        raise ValueError(f"parameters of {name!r} must be a mapping, got {params!r}")
    unknown = sorted(set(params) - set(cls.DEFAULTS))
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {unknown} for policy {name!r}; "
            f"valid: {', '.join(sorted(cls.DEFAULTS)) or '(none)'}"
        )
    missing = sorted(set(cls.DEFAULTS) - set(params))
    if strict and missing:
        raise ValueError(f"missing parameter(s) {missing} for policy {name!r}")
    return cls.check_params({**cls.DEFAULTS, **params})


def make_policy(name: str, params: Mapping[str, Any] | None = None) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; known: {', '.join(POLICY_NAMES)}")
    return POLICIES[name](**dict(params or {}))
