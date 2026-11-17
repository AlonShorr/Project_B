"""A* waypoint planner for the simulator grid — drop-in replacement for plan_rrt.

    from astar_planner import plan_astar, path_cells_to_world

    result = plan_astar(
        wall_map=wall_map,
        start=(localized_r, localized_c, localized_theta),
        goal=(goal_r, goal_c),
    )
    if result["success"]:
        path_cells = result["path"]
        path_world = path_cells_to_world(path_cells)
    else:
        print("A* failed:", result["debug"]["reason"])
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Optional

from rrt_planner import (
    FORWARD,
    TURN_AROUND,
    TURN_LEFT,
    TURN_RIGHT,
    _action_cost,
    _make_failure,
    _map_shape,
    _success_result,
    apply_internal_action,
    is_state_valid,
    normalize_state,
    path_cells_to_world,
    state_path_to_cell_path,
)

_ACTIONS = (FORWARD, TURN_LEFT, TURN_RIGHT, TURN_AROUND)


def _heuristic(state: tuple[int, int, int], goal_cell: tuple[int, int]) -> float:
    """Manhattan distance — admissible since each forward step costs 1.0 and moves one cell."""
    return abs(state[0] - goal_cell[0]) + abs(state[1] - goal_cell[1])


def plan_astar(
    wall_map: Any,
    start: tuple[int, ...],
    goal: tuple[int, ...],
    start_theta: Optional[int] = None,
    goal_theta: Optional[int] = None,
    return_debug: bool = True,
) -> dict[str, Any]:
    """Compute an optimal waypoint path from start to goal using A*.

    Searches over (r, c, theta) states with turn and forward costs from _action_cost,
    guaranteeing the least-cost path. Returns the same dict format as plan_rrt:
      result["path"]            — list of (r, c) cells, start cell excluded
      result["path_with_theta"] — list of (r, c, theta) states, start excluded
      result["success"]         — bool
      result["debug"]           — diagnostics dict
    """
    try:
        start_state = normalize_state(start, start_theta, default_theta=0)
        goal_default_theta = goal_theta if goal_theta is not None else start_state[2]
        goal_state = normalize_state(goal, goal_theta, default_theta=goal_default_theta)
    except (TypeError, ValueError) as exc:
        return _make_failure(str(exc), 0, 0)

    if not is_state_valid(wall_map, start_state):
        return _make_failure("start is outside the map", 0, 0)
    if not is_state_valid(wall_map, goal_state):
        return _make_failure("goal is outside the map", 0, 0)

    goal_cell = goal_state[:2]
    rows, cols = _map_shape(wall_map)

    g_score: dict[tuple[int, int, int], float] = {start_state: 0.0}
    came_from: dict[tuple[int, int, int], Optional[tuple[int, int, int]]] = {
        start_state: None
    }

    # (f_cost, g_cost, state) — all fields are natively comparable
    open_heap: list = [(_heuristic(start_state, goal_cell), 0.0, start_state)]
    closed: set[tuple[int, int, int]] = set()
    nodes_expanded = 0

    while open_heap:
        f, g, state = heapq.heappop(open_heap)

        if state in closed:
            continue
        closed.add(state)
        nodes_expanded += 1

        cell_match = state[:2] == goal_cell
        theta_match = goal_theta is None or state[2] == goal_state[2]
        if cell_match and theta_match:
            state_path: list[tuple[int, int, int]] = []
            s: Optional[tuple[int, int, int]] = state
            while s is not None:
                state_path.append(s)
                s = came_from[s]
            state_path.reverse()

            path = state_path_to_cell_path(state_path)
            debug = {
                "reason": "goal reached",
                "nodes_expanded": nodes_expanded,
                "iterations": nodes_expanded,
                "num_nodes": nodes_expanded,
                "path_source": "astar",
                "map_shape": (rows, cols, 4),
                "start_state": start_state,
                "goal_state": goal_state,
                "goal_cell": goal_cell,
            }
            return _success_result(
                wall_map, start_state, goal_state, goal_theta, path, state_path, debug
            )

        for action in _ACTIONS:
            nxt = apply_internal_action(wall_map, state, action)
            if nxt is None or nxt in closed:
                continue
            new_g = g + _action_cost(action)
            if new_g < g_score.get(nxt, math.inf):
                g_score[nxt] = new_g
                came_from[nxt] = state
                heapq.heappush(
                    open_heap,
                    (new_g + _heuristic(nxt, goal_cell), new_g, nxt),
                )

    return _make_failure(
        "no path exists",
        nodes_expanded,
        nodes_expanded,
        {
            "map_shape": (rows, cols, 4),
            "start_state": start_state,
            "goal_state": goal_state,
            "goal_cell": goal_cell,
        },
    )
