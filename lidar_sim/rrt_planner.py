"""Standalone RRT-style waypoint planner for the simulator grid.

The official planner output is a list of target cells, not low-level robot
commands. Internal actions are kept only for expansion and debug traces.

Integration sketch:

    from rrt_planner import plan_rrt, path_cells_to_world

    result = plan_rrt(
        wall_map=wall_map,
        start=(localized_r, localized_c, localized_theta),
        goal=(goal_r, goal_c),
    )

    if result["success"]:
        path_cells = result["path"]
        path_world = path_cells_to_world(path_cells)

        # Segev's planning block can insert path_cells or path_world into the
        # target array. The real control layer should receive the next target
        # position from this array.
    else:
        print("RRT failed:", result["debug"]["reason"])
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Optional

WALL_UP = 0
WALL_DOWN = 1
WALL_LEFT = 2
WALL_RIGHT = 3

TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
TURN_AROUND = "TURN_AROUND"
FORWARD = "FORWARD"

INTERNAL_ACTIONS = (FORWARD, TURN_LEFT, TURN_RIGHT, TURN_AROUND)

# The Pygame simulator uses 0 degrees as east and increments clockwise:
# 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE.
THETA_TO_MOVE = {
    0: (0, 1, WALL_RIGHT),
    2: (1, 0, WALL_DOWN),
    4: (0, -1, WALL_LEFT),
    6: (-1, 0, WALL_UP),
}


@dataclass(frozen=True)
class _Node:
    state: tuple[int, int, int]
    parent: Optional[int]
    action_from_parent: Optional[str]
    cost: float


def _empty_rrt_stats() -> dict[str, Any]:
    return {
        "duplicate_states_rejected": 0,
        "invalid_forward_moves": 0,
        "invalid_wall_collisions": 0,
        "invalid_out_of_bounds": 0,
        "diagonal_forward_blocks": 0,
        "goal_samples": 0,
        "successful_expansions": 0,
    }


def _map_shape(wall_map: Any) -> tuple[int, int]:
    if not hasattr(wall_map, "shape"):
        rows = len(wall_map)
        cols = len(wall_map[0]) if rows else 0
        return rows, cols
    if len(wall_map.shape) != 3 or wall_map.shape[2] < 4:
        raise ValueError("wall_map must have shape (rows, cols, 4)")
    return int(wall_map.shape[0]), int(wall_map.shape[1])


def normalize_state(
    pose: tuple[int, ...],
    theta: Optional[int] = None,
    default_theta: int = 0,
) -> tuple[int, int, int]:
    """Return a normalized ``(r, c, theta_idx)`` state."""
    if len(pose) == 3:
        r, c, theta_idx = pose
    elif len(pose) == 2:
        r, c = pose
        theta_idx = default_theta if theta is None else theta
    else:
        raise ValueError("pose must be (r, c) or (r, c, theta_idx)")
    return int(r), int(c), int(theta_idx) % 8


def is_state_valid(wall_map: Any, state: tuple[int, int, int]) -> bool:
    rows, cols = _map_shape(wall_map)
    r, c, theta_idx = state
    return 0 <= r < rows and 0 <= c < cols and 0 <= theta_idx < 8


def theta_to_cardinal_move(theta_idx: int) -> Optional[tuple[int, int, int]]:
    """Return ``(dr, dc, wall_index)`` for cardinal headings.

    Diagonal headings return ``None`` so the planner never moves diagonally
    through cell corners.
    """
    return THETA_TO_MOVE.get(theta_idx % 8)


def can_move_forward(wall_map: Any, state: tuple[int, int, int]) -> bool:
    if not is_state_valid(wall_map, state):
        return False

    move = theta_to_cardinal_move(state[2])
    if move is None:
        return False

    rows, cols = _map_shape(wall_map)
    r, c, _ = state
    dr, dc, wall_idx = move
    nr, nc = r + dr, c + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return False
    return int(wall_map[r][c][wall_idx]) == 1


def _forward_block_reason(wall_map: Any, state: tuple[int, int, int]) -> str:
    if not is_state_valid(wall_map, state):
        return "invalid_state"

    move = theta_to_cardinal_move(state[2])
    if move is None:
        return "diagonal_heading"

    rows, cols = _map_shape(wall_map)
    r, c, _ = state
    dr, dc, wall_idx = move
    nr, nc = r + dr, c + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return "outside_map"
    if int(wall_map[r][c][wall_idx]) != 1:
        return "wall"
    return "open"


def apply_internal_action(
    wall_map: Any,
    state: tuple[int, int, int],
    action: str,
) -> Optional[tuple[int, int, int]]:
    r, c, theta_idx = state
    if action == TURN_LEFT:
        return r, c, (theta_idx - 1) % 8
    if action == TURN_RIGHT:
        return r, c, (theta_idx + 1) % 8
    if action == TURN_AROUND:
        return r, c, (theta_idx + 4) % 8
    if action == FORWARD:
        if not can_move_forward(wall_map, state):
            return None
        dr, dc, _ = theta_to_cardinal_move(theta_idx)
        return r + dr, c + dc, theta_idx
    raise ValueError(f"unknown internal action: {action}")


def _theta_delta(a: int, b: int) -> int:
    diff = abs((a - b) % 8)
    return min(diff, 8 - diff)


def state_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Distance used by nearest-node selection."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) + 0.15 * _theta_delta(a[2], b[2])


def sample_state(wall_map: Any, rng: random.Random) -> tuple[int, int, int]:
    rows, cols = _map_shape(wall_map)
    return rng.randrange(rows), rng.randrange(cols), rng.randrange(8)


def nearest_node(nodes: list[_Node], target: tuple[int, int, int]) -> int:
    return min(range(len(nodes)), key=lambda idx: state_distance(nodes[idx].state, target))


def reconstruct_state_path(nodes: list[_Node], node_idx: int) -> list[tuple[int, int, int]]:
    path = []
    while node_idx is not None:
        node = nodes[node_idx]
        path.append(node.state)
        node_idx = node.parent
    path.reverse()
    return path


def _reconstruct_actions(nodes: list[_Node], node_idx: int) -> list[str]:
    actions = []
    while node_idx is not None:
        node = nodes[node_idx]
        if node.action_from_parent is not None:
            actions.append(node.action_from_parent)
        node_idx = node.parent
    actions.reverse()
    return actions


def state_path_to_cell_path(state_path: Iterable[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Convert states to official waypoint cells, excluding the start cell."""
    cells = [(r, c) for r, c, _ in state_path]
    if not cells:
        return []

    compact = [cells[0]]
    for cell in cells[1:]:
        if cell != compact[-1]:
            compact.append(cell)
    return compact[1:]


def state_path_to_debug_actions(state_path: list[tuple[int, int, int]]) -> list[str]:
    actions = []
    for prev, nxt in zip(state_path, state_path[1:]):
        if prev[:2] != nxt[:2]:
            actions.append(FORWARD)
            continue
        delta = (nxt[2] - prev[2]) % 8
        if delta == 1:
            actions.append(TURN_RIGHT)
        elif delta == 7:
            actions.append(TURN_LEFT)
        elif delta == 4:
            actions.append(TURN_AROUND)
        else:
            actions.append(f"TURN_{delta}_STEPS")
    return actions


def _movement_direction(a: tuple[int, int], b: tuple[int, int]) -> Optional[int]:
    dr, dc = b[0] - a[0], b[1] - a[1]
    for theta_idx, (tdr, tdc, _) in THETA_TO_MOVE.items():
        if (dr, dc) == (tdr, tdc):
            return theta_idx
    return None


def _path_turns(cells: list[tuple[int, int]]) -> int:
    turns = 0
    last_dir = None
    for a, b in zip(cells, cells[1:]):
        direction = _movement_direction(a, b)
        if last_dir is not None and direction is not None and direction != last_dir:
            turns += 1
        if direction is not None:
            last_dir = direction
    return turns


def _cell_path_cost(cells: list[tuple[int, int]], turn_penalty: float = 0.25) -> float:
    return max(0, len(cells) - 1) + turn_penalty * _path_turns(cells)


def _step_is_valid(wall_map: Any, a: tuple[int, int], b: tuple[int, int]) -> bool:
    direction = _movement_direction(a, b)
    if direction is None:
        return False
    state = (a[0], a[1], direction)
    return can_move_forward(wall_map, state)


def bfs_shortest_path_cells(
    wall_map: Any,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """Return shortest cells from start to goal, inclusive, or None."""
    rows, cols = _map_shape(wall_map)
    start_cell = (int(start_cell[0]), int(start_cell[1]))
    goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
    if not (0 <= start_cell[0] < rows and 0 <= start_cell[1] < cols):
        return None
    if not (0 <= goal_cell[0] < rows and 0 <= goal_cell[1] < cols):
        return None

    queue = deque([start_cell])
    parent = {start_cell: None}
    while queue:
        cell = queue.popleft()
        if cell == goal_cell:
            break
        r, c = cell
        for theta_idx in (0, 2, 4, 6):
            if not can_move_forward(wall_map, (r, c, theta_idx)):
                continue
            dr, dc, _ = THETA_TO_MOVE[theta_idx]
            nxt = (r + dr, c + dc)
            if nxt in parent:
                continue
            parent[nxt] = cell
            queue.append(nxt)

    if goal_cell not in parent:
        return None

    path = []
    cell = goal_cell
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path


def validate_path(
    wall_map: Any,
    start: tuple[int, ...],
    goal: tuple[int, ...],
    path: list[tuple[int, int]],
) -> tuple[bool, str]:
    """Validate official waypoint output against map walls."""
    start_cell = (int(start[0]), int(start[1]))
    goal_cell = (int(goal[0]), int(goal[1]))

    if start_cell == goal_cell and not path:
        return True, "already at goal"
    if not path:
        return False, "path is empty"
    if path[-1] != goal_cell:
        return False, "final cell does not equal goal"

    prev = start_cell
    rows, cols = _map_shape(wall_map)
    for cell in path:
        r, c = cell
        if not (0 <= r < rows and 0 <= c < cols):
            return False, f"cell {cell} is outside the map"
        if abs(r - prev[0]) + abs(c - prev[1]) != 1:
            return False, f"cells {prev} and {cell} are not neighbors"
        if not _step_is_valid(wall_map, prev, cell):
            return False, f"step {prev} -> {cell} crosses a wall"
        prev = cell
    return True, "valid"


def _action_cost(action: str) -> float:
    if action == FORWARD:
        return 1.0
    if action == TURN_AROUND:
        return 0.5
    return 0.25


def _shortest_cell_path(
    wall_map: Any,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return shortest cell waypoints excluding ``start_cell``."""
    path = bfs_shortest_path_cells(wall_map, start_cell, goal_cell)
    return [] if path is None else path[1:]


def _cell_path_to_state_path(
    start_state: tuple[int, int, int],
    path: list[tuple[int, int]],
    final_theta: Optional[int] = None,
) -> list[tuple[int, int, int]]:
    state_path = [start_state]
    current = start_state
    for cell in path:
        target_theta = _movement_direction(current[:2], cell)
        if target_theta is None:
            raise ValueError(f"cells {current[:2]} and {cell} are not neighbors")
        while current[2] != target_theta:
            right_steps = (target_theta - current[2]) % 8
            if right_steps == 4:
                current = (current[0], current[1], (current[2] + 4) % 8)
            elif right_steps < 4:
                current = (current[0], current[1], (current[2] + 1) % 8)
            else:
                current = (current[0], current[1], (current[2] - 1) % 8)
            state_path.append(current)
        current = (cell[0], cell[1], current[2])
        state_path.append(current)
    if final_theta is not None:
        final_theta = final_theta % 8
        while current[2] != final_theta:
            right_steps = (final_theta - current[2]) % 8
            if right_steps == 4:
                current = (current[0], current[1], (current[2] + 4) % 8)
            elif right_steps < 4:
                current = (current[0], current[1], (current[2] + 1) % 8)
            else:
                current = (current[0], current[1], (current[2] - 1) % 8)
            state_path.append(current)
    return state_path


def _best_action_toward(
    wall_map: Any,
    state: tuple[int, int, int],
    target: tuple[int, int, int],
    rng: random.Random,
) -> list[tuple[str, tuple[int, int, int]]]:
    candidates = []
    for action in INTERNAL_ACTIONS:
        nxt = apply_internal_action(wall_map, state, action)
        if nxt is None:
            continue
        score = state_distance(nxt, target) + 0.03 * rng.random()
        candidates.append((score, action, nxt))

    candidates.sort(key=lambda item: item[0])
    return [(action, nxt) for _, action, nxt in candidates]


def _ranked_child_actions(
    wall_map: Any,
    state: tuple[int, int, int],
    target: tuple[int, int, int],
    rng: random.Random,
    stats: dict[str, Any],
) -> list[tuple[str, tuple[int, int, int]]]:
    candidates = []
    for action in INTERNAL_ACTIONS:
        if action == FORWARD:
            reason = _forward_block_reason(wall_map, state)
            if reason != "open":
                stats["invalid_forward_moves"] += 1
                if reason == "diagonal_heading":
                    stats["diagonal_forward_blocks"] += 1
                elif reason == "outside_map":
                    stats["invalid_out_of_bounds"] += 1
                elif reason == "wall":
                    stats["invalid_wall_collisions"] += 1
                continue

        nxt = apply_internal_action(wall_map, state, action)
        if nxt is None:
            continue
        score = state_distance(nxt, target) + 0.001 * rng.random()
        candidates.append((score, action, nxt))

    candidates.sort(key=lambda item: item[0])
    return [(action, nxt) for _, action, nxt in candidates]


def _make_failure(
    reason: str,
    iterations: int,
    num_nodes: int,
    extra_debug: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    debug = {
        "reason": reason,
        "iterations": iterations,
        "num_nodes": num_nodes,
    }
    if extra_debug:
        debug.update(extra_debug)
    return {
        "success": False,
        "path": [],
        "path_with_theta": [],
        "debug": debug,
    }


def _closest_node_to_goal(
    nodes: list[_Node],
    goal_cell: tuple[int, int],
) -> tuple[tuple[int, int, int], float]:
    closest = min(nodes, key=lambda node: math.hypot(node.state[0] - goal_cell[0], node.state[1] - goal_cell[1]))
    dist = math.hypot(closest.state[0] - goal_cell[0], closest.state[1] - goal_cell[1])
    return closest.state, dist


def _success_result(
    wall_map: Any,
    start_state: tuple[int, int, int],
    goal_state: tuple[int, int, int],
    goal_theta: Optional[int],
    path: list[tuple[int, int]],
    state_path: list[tuple[int, int, int]],
    debug: dict[str, Any],
) -> dict[str, Any]:
    valid, reason = validate_path(wall_map, start_state, goal_state, path)
    if not valid:
        return _make_failure(f"constructed path failed validation: {reason}", debug.get("iterations", 0), debug.get("num_nodes", 0), debug)

    debug_actions = state_path_to_debug_actions(state_path)
    debug["debug_actions"] = debug_actions
    debug["path_cost"] = _cell_path_cost([start_state[:2]] + path)
    return {
        "success": True,
        "path": path,
        "path_with_theta": state_path[1:],
        "debug": debug,
    }


def plan_rrt(
    wall_map: Any,
    start: tuple[int, ...],
    goal: tuple[int, ...],
    start_theta: Optional[int] = None,
    goal_theta: Optional[int] = None,
    max_iter: int = 2000,
    goal_sample_rate: float = 0.15,
    random_seed: Optional[int] = None,
    return_debug: bool = True,
    fallback_to_bfs: bool = False,
) -> dict[str, Any]:
    """Compute a waypoint path from localized start to goal.

    Returns a dict whose official control output is ``result["path"]``:
    ``[(r1, c1), (r2, c2), ...]``. The current robot cell is intentionally
    omitted, so ``path[0]`` is the next target after the current position.
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

    rows, cols = _map_shape(wall_map)
    bfs_path = bfs_shortest_path_cells(wall_map, start_state[:2], goal_state[:2])
    bfs_reachable = bfs_path is not None
    bfs_length = None if bfs_path is None else max(0, len(bfs_path) - 1)

    rng = random.Random(random_seed)
    nodes = [_Node(start_state, None, None, 0.0)]
    seen = {start_state}
    tree_edges = []
    best_goal_idx = None
    best_goal_cost = math.inf
    goal_cell = goal_state[:2]
    stats = _empty_rrt_stats()

    for iteration in range(1, max_iter + 1):
        if rng.random() < goal_sample_rate:
            target = goal_state
            stats["goal_samples"] += 1
        else:
            target = sample_state(wall_map, rng)

        ranked_node_indices = sorted(range(len(nodes)), key=lambda idx: state_distance(nodes[idx].state, target))[:32]
        next_step = None
        near_idx = None
        for candidate_idx in ranked_node_indices:
            ranked_actions = _ranked_child_actions(wall_map, nodes[candidate_idx].state, target, rng, stats)
            for action, state in ranked_actions:
                if state in seen:
                    stats["duplicate_states_rejected"] += 1
                    continue
                next_step = (action, state)
                near_idx = candidate_idx
                break
            if next_step is not None:
                break

        if next_step is None:
            continue
        action, new_state = next_step

        new_cost = nodes[near_idx].cost + _action_cost(action)
        nodes.append(_Node(new_state, near_idx, action, new_cost))
        new_idx = len(nodes) - 1
        seen.add(new_state)
        stats["successful_expansions"] += 1
        if return_debug:
            tree_edges.append((nodes[near_idx].state, new_state))

        cell_matches = new_state[:2] == goal_cell
        theta_matches = goal_theta is None or new_state[2] == goal_state[2]
        if cell_matches and theta_matches and new_cost < best_goal_cost:
            best_goal_idx = new_idx
            best_goal_cost = new_cost

        # Keep a little time for a shorter route, but return promptly once a
        # valid route has had a chance to appear.
        if best_goal_idx is not None and iteration > min(80, max_iter // 5):
            break

    if best_goal_idx is None:
        closest_state, closest_dist = _closest_node_to_goal(nodes, goal_cell)
        failure_debug = {
            "map_shape": (rows, cols, 4),
            "start_state": start_state,
            "goal_state": goal_state,
            "goal_cell": goal_cell,
            "unique_states_in_tree": len(seen),
            "total_possible_states": rows * cols * 8,
            "closest_node_to_goal": closest_state,
            "closest_distance_to_goal": closest_dist,
            "tree_edges": tree_edges if return_debug else [],
            "bfs_reachable": bfs_reachable,
            "bfs_shortest_path_length": bfs_length,
            **stats,
        }
        if fallback_to_bfs and bfs_path is not None:
            path = bfs_path[1:]
            state_path = _cell_path_to_state_path(start_state, path, goal_theta)
            failure_debug.update(
                {
                    "reason": "RRT failed, BFS fallback used",
                    "path_source": "bfs_fallback",
                    "iterations": max_iter,
                    "num_nodes": len(nodes),
                    "raw_rrt_path": [],
                    "raw_rrt_path_with_theta": [],
                }
            )
            return _success_result(wall_map, start_state, goal_state, goal_theta, path, state_path, failure_debug)

        return _make_failure("max_iter reached without connecting to goal", max_iter, len(nodes), failure_debug)

    state_path = reconstruct_state_path(nodes, best_goal_idx)
    path = state_path_to_cell_path(state_path)
    raw_rrt_path = path
    raw_rrt_path_with_theta = state_path[1:]
    valid, reason = validate_path(wall_map, start_state, goal_state, path)
    if not valid:
        return _make_failure(f"constructed path failed validation: {reason}", max_iter, len(nodes))

    shortest_path = _shortest_cell_path(wall_map, start_state[:2], goal_cell)
    path_source = "rrt_tree"
    if shortest_path and _cell_path_cost([start_state[:2]] + shortest_path) <= _cell_path_cost([start_state[:2]] + path):
        path = shortest_path
        state_path = _cell_path_to_state_path(start_state, path, goal_theta)
        path_source = "shortest_valid_postprocess"

    debug = {
        "iterations": iteration,
        "num_nodes": len(nodes),
        "reason": "goal reached",
        "tree_edges": tree_edges if return_debug else [],
        "path_source": path_source,
        "raw_rrt_path": raw_rrt_path,
        "raw_rrt_path_with_theta": raw_rrt_path_with_theta,
        "map_shape": (rows, cols, 4),
        "start_state": start_state,
        "goal_state": goal_state,
        "goal_cell": goal_cell,
        "unique_states_in_tree": len(seen),
        "total_possible_states": rows * cols * 8,
        "bfs_reachable": bfs_reachable,
        "bfs_shortest_path_length": bfs_length,
        **stats,
    }
    return _success_result(wall_map, start_state, goal_state, goal_theta, path, state_path, debug)


def cell_to_world(r: int, c: int, cell_size_m: float = 0.18) -> tuple[float, float]:
    """Convert a grid cell center to physical ``(x_m, y_m)`` coordinates."""
    x = (int(c) + 0.5) * cell_size_m
    y = (int(r) + 0.5) * cell_size_m
    return x, y


def path_cells_to_world(
    path: Iterable[tuple[int, int]],
    cell_size_m: float = 0.18,
) -> list[tuple[float, float]]:
    return [cell_to_world(r, c, cell_size_m) for r, c in path]


def _wall_signature(wall_map: Any, r: int, c: int) -> tuple[int, int, int, int]:
    return tuple(int(wall_map[r][c][idx]) for idx in range(4))


def route_observability_score(path: list[tuple[int, int]], wall_map: Any) -> float:
    """Score paths higher when they visit distinctive local wall patterns."""
    rows, cols = _map_shape(wall_map)
    signatures = Counter(_wall_signature(wall_map, r, c) for r in range(rows) for c in range(cols))
    if not path:
        return 0.0

    score = 0.0
    last_signature = None
    for r, c in path:
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        signature = _wall_signature(wall_map, r, c)
        score += 1.0 / signatures[signature]
        if last_signature is not None and signature != last_signature:
            score += 0.25
        last_signature = signature
    return score


def _reachable_cells(
    wall_map: Any,
    start_cell: tuple[int, int],
    max_depth: int = 6,
) -> list[tuple[int, int, int]]:
    rows, cols = _map_shape(wall_map)
    if not (0 <= start_cell[0] < rows and 0 <= start_cell[1] < cols):
        return []

    visited = {start_cell}
    queue = deque([(start_cell[0], start_cell[1], 0)])
    result = []
    while queue:
        r, c, depth = queue.popleft()
        result.append((r, c, depth))
        if depth >= max_depth:
            continue
        for theta_idx in (0, 2, 4, 6):
            dr, dc, _ = THETA_TO_MOVE[theta_idx]
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if can_move_forward(wall_map, (r, c, theta_idx)):
                visited.add((nr, nc))
                queue.append((nr, nc, depth + 1))
    return result


def choose_localization_improving_waypoint(
    belief: Any,
    wall_map: Any,
    current_position: tuple[int, ...],
    expected_data: Any = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Suggest a nearby waypoint likely to improve localization confidence."""
    del expected_data  # Reserved for Segev's richer loop when it is ready.
    current = (int(current_position[0]), int(current_position[1]))
    candidates = _reachable_cells(wall_map, current, max_depth=max(2, top_k + 1))
    if not candidates:
        return {"success": False, "waypoint": None, "reason": "current position is invalid or isolated"}

    rows, cols = _map_shape(wall_map)
    signatures = Counter(_wall_signature(wall_map, r, c) for r in range(rows) for c in range(cols))

    scored = []
    for r, c, depth in candidates:
        if (r, c) == current:
            continue
        signature = _wall_signature(wall_map, r, c)
        rarity = 1.0 / signatures[signature]
        local_variety = 0.0
        for theta_idx in (0, 2, 4, 6):
            if can_move_forward(wall_map, (r, c, theta_idx)):
                dr, dc, _ = THETA_TO_MOVE[theta_idx]
                local_variety += _wall_signature(wall_map, r + dr, c + dc) != signature
        belief_penalty = 0.0
        try:
            belief_penalty = float(belief[r][c])
        except (TypeError, IndexError, KeyError):
            pass
        score = rarity + 0.15 * local_variety - 0.05 * depth - 0.1 * belief_penalty
        scored.append((score, depth, (r, c)))

    if not scored:
        return {"success": False, "waypoint": None, "reason": "no reachable waypoint candidates"}

    scored.sort(key=lambda item: (-item[0], item[1]))
    score, depth, waypoint = scored[0]
    return {
        "success": True,
        "waypoint": waypoint,
        "score": score,
        "reason": f"selected reachable distinctive cell at depth {depth}",
    }
