from collections import deque

import numpy as np

from config import MAP_COLS, MAP_ROWS, WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP


_DIRS = (
    (-1, 0, WALL_UP, WALL_DOWN),
    (1, 0, WALL_DOWN, WALL_UP),
    (0, -1, WALL_LEFT, WALL_RIGHT),
    (0, 1, WALL_RIGHT, WALL_LEFT),
)


def _base_map():
    wall_map = np.ones((MAP_ROWS, MAP_COLS, 4), dtype=int)
    wall_map[0, :, WALL_UP] = 0
    wall_map[-1, :, WALL_DOWN] = 0
    wall_map[:, 0, WALL_LEFT] = 0
    wall_map[:, -1, WALL_RIGHT] = 0
    return wall_map


def _set_edge(wall_map, r, c, wall, value):
    wall_map[r, c, wall] = value
    if wall == WALL_UP and r > 0:
        wall_map[r - 1, c, WALL_DOWN] = value
    elif wall == WALL_DOWN and r < MAP_ROWS - 1:
        wall_map[r + 1, c, WALL_UP] = value
    elif wall == WALL_LEFT and c > 0:
        wall_map[r, c - 1, WALL_RIGHT] = value
    elif wall == WALL_RIGHT and c < MAP_COLS - 1:
        wall_map[r, c + 1, WALL_LEFT] = value


def _set_h_wall(wall_map, r, c):
    _set_edge(wall_map, r, c, WALL_DOWN, 0)


def _set_v_wall(wall_map, r, c):
    _set_edge(wall_map, r, c, WALL_RIGHT, 0)


def _set_h_segment(wall_map, r, c_start, c_end, gaps=()):
    for c in range(c_start, c_end + 1):
        if c not in gaps:
            _set_h_wall(wall_map, r, c)


def _set_v_segment(wall_map, c, r_start, r_end, gaps=()):
    for r in range(r_start, r_end + 1):
        if r not in gaps:
            _set_v_wall(wall_map, r, c)


def _closed_map():
    return np.zeros((MAP_ROWS, MAP_COLS, 4), dtype=int)


def _open_edge(wall_map, r, c, wall):
    _set_edge(wall_map, r, c, wall, 1)


def _open_between(wall_map, cell, neighbor):
    r, c = cell
    nr, nc = neighbor
    if nr == r - 1 and nc == c:
        _open_edge(wall_map, r, c, WALL_UP)
    elif nr == r + 1 and nc == c:
        _open_edge(wall_map, r, c, WALL_DOWN)
    elif nr == r and nc == c - 1:
        _open_edge(wall_map, r, c, WALL_LEFT)
    elif nr == r and nc == c + 1:
        _open_edge(wall_map, r, c, WALL_RIGHT)


def _dense_photo_maze(seed, start, goal, loop_edges=()):
    rng = np.random.default_rng(seed)
    wall_map = _closed_map()
    visited = {start}
    stack = [start]

    while stack:
        r, c = stack[-1]
        candidates = []
        for dr, dc, _wall, _opposite in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < MAP_ROWS and 0 <= nc < MAP_COLS and (nr, nc) not in visited:
                candidates.append((nr, nc))
        if not candidates:
            stack.pop()
            continue
        nr, nc = candidates[int(rng.integers(len(candidates)))]
        _open_between(wall_map, (r, c), (nr, nc))
        visited.add((nr, nc))
        stack.append((nr, nc))

    for cell, neighbor in loop_edges:
        _open_between(wall_map, cell, neighbor)
    return wall_map, start, goal


def _generate_easy_training_maze():
    wall_map = _base_map()
    for r in range(1, MAP_ROWS - 1):
        if r not in (4, 9):
            _set_v_wall(wall_map, r, 5)
    for c in range(5, MAP_COLS - 1):
        if c not in (8, 14):
            _set_h_wall(wall_map, 6, c)
    for r in range(2, 7):
        if r != 3:
            _set_v_wall(wall_map, r, 12)
    for c in range(2, 12):
        if c != 6:
            _set_h_wall(wall_map, 10, c)
    for r in range(9, MAP_ROWS - 2):
        if r != 12:
            _set_v_wall(wall_map, r, 15)
    return wall_map, (1, 1), (MAP_ROWS - 2, MAP_COLS - 2)


def _generate_medium_crossroads():
    wall_map = _base_map()
    for r in range(1, MAP_ROWS - 1):
        if r not in (3, 6, 10):
            _set_v_wall(wall_map, r, 6)
        if r not in (4, 9):
            _set_v_wall(wall_map, r, 14)
        if r not in (2, 8, 12):
            _set_v_wall(wall_map, r, 10)
    for c in range(2, MAP_COLS - 1):
        if c not in (4, 11, 16):
            _set_h_wall(wall_map, 7, c)
        if c not in (5, 13, 18):
            _set_h_wall(wall_map, 4, c)
        if c not in (3, 9, 15):
            _set_h_wall(wall_map, 11, c)
    for c in range(2, MAP_COLS - 2, 4):
        _set_h_wall(wall_map, 3, c)
    for r in (2, 5, 8, 12):
        for c in (2, 17):
            _set_v_wall(wall_map, r, c)
    return wall_map, (1, 1), (MAP_ROWS - 2, MAP_COLS - 2)


def _generate_medium_rooms():
    wall_map = _base_map()
    for c in range(1, MAP_COLS - 1):
        if c not in (3, 9, 15):
            _set_h_wall(wall_map, 4, c)
        if c not in (5, 12, 18):
            _set_h_wall(wall_map, 9, c)
    for r in range(1, MAP_ROWS - 1):
        if r not in (2, 7, 12):
            _set_v_wall(wall_map, r, 5)
        if r not in (4, 8, 11):
            _set_v_wall(wall_map, r, 13)
        if r not in (3, 6, 10, 13):
            _set_v_wall(wall_map, r, 9)
    for c in range(2, MAP_COLS - 2, 3):
        _set_h_wall(wall_map, 2, c)
        _set_h_wall(wall_map, 12, c)
    for r in (2, 6, 10):
        for c in (2, 16):
            _set_v_wall(wall_map, r, c)
    return wall_map, (2, 2), (MAP_ROWS - 3, MAP_COLS - 3)


def _generate_hard_serpentine():
    wall_map = _base_map()
    barrier_rows = (1, 3, 5, 7, 9, 11, 13)
    for idx, r in enumerate(barrier_rows):
        gap = 1 if idx % 2 == 0 else MAP_COLS - 2
        for c in range(0, MAP_COLS - 1):
            if c != gap:
                _set_h_wall(wall_map, r, c)
    for r in range(2, MAP_ROWS - 2, 4):
        _set_v_wall(wall_map, r, 8)
        _set_v_wall(wall_map, r, 14)
    for r in range(2, MAP_ROWS - 2, 2):
        for c in (3, 6, 11, 17):
            if (r + c) % 3 != 0:
                _set_v_wall(wall_map, r, c)
    for c in range(4, MAP_COLS - 4, 4):
        _set_h_wall(wall_map, 2, c)
        _set_h_wall(wall_map, 12, c)
    return wall_map, (0, 0), (MAP_ROWS - 1, MAP_COLS - 1)


def _generate_hard_chambers():
    wall_map = _base_map()
    for c in range(1, MAP_COLS - 1):
        if c not in (2, 10, 17):
            _set_h_wall(wall_map, 2, c)
        if c not in (6, 13):
            _set_h_wall(wall_map, 5, c)
        if c not in (3, 11, 18):
            _set_h_wall(wall_map, 8, c)
        if c not in (8, 15):
            _set_h_wall(wall_map, 11, c)
    for r in range(1, MAP_ROWS - 1):
        if r not in (1, 4, 9, 13):
            _set_v_wall(wall_map, r, 3)
        if r not in (3, 6, 12):
            _set_v_wall(wall_map, r, 9)
        if r not in (2, 7, 10):
            _set_v_wall(wall_map, r, 16)
        if r not in (5, 8, 13):
            _set_v_wall(wall_map, r, 6)
        if r not in (1, 6, 11):
            _set_v_wall(wall_map, r, 12)
    for c in range(2, MAP_COLS - 2, 2):
        if c not in (8, 14):
            _set_h_wall(wall_map, 4, c)
        if c not in (4, 10, 18):
            _set_h_wall(wall_map, 10, c)
    for r in (3, 7, 12):
        for c in (1, 18):
            _set_v_wall(wall_map, r, c)
    return wall_map, (1, 1), (MAP_ROWS - 2, MAP_COLS - 2)


def _generate_photo_grid_a():
    loops = (
        ((7, 12), (7, 13)), ((6, 12), (6, 13)), ((6, 1), (6, 2)),
        ((4, 3), (4, 4)), ((8, 5), (8, 6)), ((10, 10), (10, 11)),
        ((11, 16), (11, 17)), ((2, 14), (2, 15)), ((13, 3), (13, 4)),
    )
    return _dense_photo_maze(601, (7, 12), (6, 1), loops)


def _generate_photo_grid_b():
    loops = (
        ((7, 13), (8, 13)), ((6, 1), (6, 2)), ((3, 7), (3, 8)),
        ((5, 15), (6, 15)), ((9, 4), (9, 5)), ((12, 12), (12, 13)),
    )
    return _dense_photo_maze(607, (7, 13), (6, 1), loops)


def _generate_photo_grid_c():
    loops = (
        ((7, 12), (7, 13)), ((5, 2), (5, 3)), ((2, 10), (2, 11)),
        ((8, 8), (9, 8)), ((12, 15), (12, 16)),
    )
    return _dense_photo_maze(613, (7, 12), (5, 2), loops)


def _generate_photo_grid_d():
    loops = (
        ((7, 12), (8, 12)), ((6, 1), (6, 2)),
    )
    return _dense_photo_maze(619, (7, 12), (6, 1), loops)


def _generate_photo_grid_e():
    loops = (
        ((7, 12), (7, 13)),
    )
    return _dense_photo_maze(631, (7, 12), (6, 1), loops)


def _neighbors(wall_map, r, c):
    for dr, dc, wall, _opposite in _DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < MAP_ROWS and 0 <= nc < MAP_COLS and wall_map[r, c, wall] == 1:
            yield nr, nc


def _reachable_cells(wall_map, start):
    queue = deque([start])
    visited = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors(wall_map, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited


def _wall_distance(wall_map, r, c, wall):
    steps = 0
    while wall_map[r, c, wall] == 1:
        if wall == WALL_UP:
            r -= 1
        elif wall == WALL_DOWN:
            r += 1
        elif wall == WALL_LEFT:
            c -= 1
        elif wall == WALL_RIGHT:
            c += 1
        steps += 1
    return steps


def _wall_distance_signature(wall_map, r, c):
    return (
        _wall_distance(wall_map, r, c, WALL_UP),
        _wall_distance(wall_map, r, c, WALL_DOWN),
        _wall_distance(wall_map, r, c, WALL_LEFT),
        _wall_distance(wall_map, r, c, WALL_RIGHT),
    )


def _shortest_path_steps(wall_map, start, goal):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        cell, steps = queue.popleft()
        if cell == goal:
            return steps
        for neighbor in _neighbors(wall_map, *cell):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, steps + 1))
    return -1


def validate_wall_map(wall_map):
    if not isinstance(wall_map, np.ndarray):
        return False
    if wall_map.shape != (MAP_ROWS, MAP_COLS, 4):
        return False
    if not np.all((wall_map == 0) | (wall_map == 1)):
        return False
    if not (
        np.all(wall_map[0, :, WALL_UP] == 0)
        and np.all(wall_map[-1, :, WALL_DOWN] == 0)
        and np.all(wall_map[:, 0, WALL_LEFT] == 0)
        and np.all(wall_map[:, -1, WALL_RIGHT] == 0)
    ):
        return False

    for r in range(MAP_ROWS):
        for c in range(MAP_COLS):
            if r > 0 and wall_map[r, c, WALL_UP] != wall_map[r - 1, c, WALL_DOWN]:
                return False
            if r < MAP_ROWS - 1 and wall_map[r, c, WALL_DOWN] != wall_map[r + 1, c, WALL_UP]:
                return False
            if c > 0 and wall_map[r, c, WALL_LEFT] != wall_map[r, c - 1, WALL_RIGHT]:
                return False
            if c < MAP_COLS - 1 and wall_map[r, c, WALL_RIGHT] != wall_map[r, c + 1, WALL_LEFT]:
                return False

    return len(_reachable_cells(wall_map, (0, 0))) == MAP_ROWS * MAP_COLS


def score_map(wall_map, start, goal):
    signatures = {}
    for r in range(MAP_ROWS):
        for c in range(MAP_COLS):
            signature = _wall_distance_signature(wall_map, r, c)
            signatures[signature] = signatures.get(signature, 0) + 1

    ambiguity_score = sum(count for count in signatures.values() if count > 1)
    shortest_path_steps = _shortest_path_steps(wall_map, start, goal)
    return {
        "ambiguity_score": ambiguity_score,
        "shortest_path_steps": shortest_path_steps,
    }


_MAP_SPECS = (
    ("Training Maze", _generate_easy_training_maze),
    ("Crossroads", _generate_medium_crossroads),
    ("Room Grid", _generate_medium_rooms),
    ("Serpentine", _generate_hard_serpentine),
    ("Chambers", _generate_hard_chambers),
    ("Photo Grid A", _generate_photo_grid_a),
    ("Photo Grid B", _generate_photo_grid_b),
    ("Photo Grid C", _generate_photo_grid_c),
    ("Photo Grid D", _generate_photo_grid_d),
    ("Photo Grid E", _generate_photo_grid_e),
)
_MAPS = None
_PHOTO_MAP_NAMES = {
    "Photo Grid A",
    "Photo Grid B",
    "Photo Grid C",
    "Photo Grid D",
    "Photo Grid E",
}


def _assign_ranked_difficulties(maps):
    ranked_maps = sorted(
        maps,
        key=lambda item: (item["ambiguity_score"] + item["shortest_path_steps"], item["shortest_path_steps"]),
    )
    for rank, map_meta in enumerate(ranked_maps):
        if rank == 0:
            difficulty = "easy"
        elif rank <= 2:
            difficulty = "medium"
        else:
            difficulty = "hard"
        map_meta["difficulty"] = difficulty


def _build_maps():
    maps = []
    for name, generator in _MAP_SPECS:
        wall_map, start, goal = generator()
        if not validate_wall_map(wall_map):
            raise ValueError(f"generated map is invalid: {name}")
        scores = score_map(wall_map, start, goal)
        if scores["shortest_path_steps"] < 0:
            raise ValueError(f"generated map has no start-to-goal path: {name}")
        maps.append(
            {
                "name": name,
                "start": start,
                "goal": goal,
                "wall_map": wall_map,
                "ambiguity_score": scores["ambiguity_score"],
                "shortest_path_steps": scores["shortest_path_steps"],
            }
        )

    photo_maps = [map_meta for map_meta in maps if map_meta["name"] in _PHOTO_MAP_NAMES]
    other_maps = [map_meta for map_meta in maps if map_meta["name"] not in _PHOTO_MAP_NAMES]
    _assign_ranked_difficulties(other_maps)
    _assign_ranked_difficulties(photo_maps)
    return maps


def get_maps():
    global _MAPS
    if _MAPS is None:
        _MAPS = _build_maps()
    return _MAPS


def get_map(index):
    return get_maps()[index]
