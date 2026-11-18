"""Maze bank for Project B (roadmap task t6): mazes of any size, stored by name.

Public API
----------
- ``MazeSpec``: a frozen record ``(name, wall_map, goal_cells, rows, cols, meta)``.
- ``load_maze(name)`` / ``list_mazes(size=None)`` / ``save_maze(spec)``: the
  stored bank under ``lidar_sim/mazes/<size>/<name>.npz`` (+ ``.txt`` drawing),
  indexed by ``lidar_sim/mazes/index.yaml``. Names look like ``medium/m10_a``.
- ``generate_maze(rows, cols, seed, loop_fraction)``: DFS recursive backtracker
  (ported from ``map_generator._dense_photo_maze``) plus random loop injection.
- ``validate_wall_map(wall_map)``: borders closed, both sides of every edge
  agree, every cell reachable. Works for any size.
- ``maze_to_wall_map(source)``: parser for the predecessor team's ASCII
  ``.maze`` format (port of ``sim.maze.Maze.from_maze_text``).
- ``ambiguity(wall_map, R=2)``: how many cells share a sensor signature, both
  for Project A's unbounded wall-distance signature and for the R-limited
  noiseless signature of the planned sensor.

Conventions (shared with the rest of lidar_sim):
- ``wall_map`` has shape ``(rows, cols, 4)``, channels WALL_UP/DOWN/LEFT/RIGHT
  (0..3), ``0 = wall`` and ``1 = open``. Both sides of an edge are stored.
- Row 0 is the top (north) row.

Unlike ``map_generator``, nothing here reads ``config.MAP_ROWS/MAP_COLS``.
"""

from __future__ import annotations

import json
import re
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP

MAZE_DIR = Path(__file__).resolve().parent / "mazes"
INDEX_PATH = MAZE_DIR / "index.yaml"
SIZES = ("small", "medium", "large")
# Required (rows, cols) for the named size classes; other size folders are free-form.
SIZE_SHAPES = {"small": (5, 5), "medium": (10, 10), "large": (16, 16)}
# One path component of a maze name: letters, digits, "_" and "-" (no "/", "..", spaces).
_NAME_PART = re.compile(r"[A-Za-z0-9_-]+")

# Reading reported by an R-limited ray when the first wall is farther than R.
NONE = -1

# (dr, dc, wall on this cell, same edge seen from the neighbour)
_DIRS = (
    (-1, 0, WALL_UP, WALL_DOWN),
    (1, 0, WALL_DOWN, WALL_UP),
    (0, -1, WALL_LEFT, WALL_RIGHT),
    (0, 1, WALL_RIGHT, WALL_LEFT),
)
_WALL_DELTA = {wall: (dr, dc) for dr, dc, wall, _ in _DIRS}
_OPPOSITE = {wall: opp for _, _, wall, opp in _DIRS}


# --------------------------------------------------------------------------
# MazeSpec
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MazeSpec:
    """One stored maze.

    ``wall_map`` is ``(rows, cols, 4)`` uint8, ``0 = wall``, ``1 = open``.
    ``goal_cells`` are ``(r, c)`` tuples. ``meta`` holds the provenance
    (generator seed and loop fraction, or the predecessor source file) and the
    ambiguity scores, as written in ``index.yaml``.
    """

    name: str
    wall_map: np.ndarray
    goal_cells: tuple[tuple[int, int], ...]
    rows: int
    cols: int
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> str:
        """Size class (``small``/``medium``/``large``), taken from the name."""
        return self.name.split("/", 1)[0] if "/" in self.name else size_class(self.rows, self.cols)

    @property
    def short_name(self) -> str:
        return self.name.split("/", 1)[-1]


def size_class(rows: int, cols: int) -> str:
    """Map a maze shape to its size class: 5x5 small, 10x10 medium, 16x16 large."""
    return {(5, 5): "small", (10, 10): "medium", (16, 16): "large"}.get(
        (rows, cols), f"{rows}x{cols}"
    )


def center_goal(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """Center cell for odd sizes, the 2x2 center block for even sizes.

    For a mixed odd/even shape the goal is the 1x2 / 2x1 center strip.
    """
    rs = (rows // 2,) if rows % 2 else (rows // 2 - 1, rows // 2)
    cs = (cols // 2,) if cols % 2 else (cols // 2 - 1, cols // 2)
    return tuple((r, c) for r in rs for c in cs)


# --------------------------------------------------------------------------
# Edge helpers (size-generic versions of map_generator's private helpers)
# --------------------------------------------------------------------------

def closed_wall_map(rows: int, cols: int) -> np.ndarray:
    """Every edge closed."""
    return np.zeros((rows, cols, 4), dtype=np.uint8)


def open_wall_map(rows: int, cols: int) -> np.ndarray:
    """Only the outer border closed."""
    wall_map = np.ones((rows, cols, 4), dtype=np.uint8)
    wall_map[0, :, WALL_UP] = 0
    wall_map[-1, :, WALL_DOWN] = 0
    wall_map[:, 0, WALL_LEFT] = 0
    wall_map[:, -1, WALL_RIGHT] = 0
    return wall_map


def set_edge(wall_map: np.ndarray, r: int, c: int, wall: int, value: int) -> None:
    """Set one edge to ``value`` (0 wall, 1 open) on both of its sides."""
    rows, cols = wall_map.shape[:2]
    wall_map[r, c, wall] = value
    dr, dc = _WALL_DELTA[wall]
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        wall_map[nr, nc, _OPPOSITE[wall]] = value


def _wall_between(cell: tuple[int, int], neighbor: tuple[int, int]) -> int:
    dr, dc = neighbor[0] - cell[0], neighbor[1] - cell[1]
    for ddr, ddc, wall, _ in _DIRS:
        if (ddr, ddc) == (dr, dc):
            return wall
    raise ValueError(f"cells {cell} and {neighbor} are not 4-adjacent")


def open_between(wall_map: np.ndarray, cell: tuple[int, int], neighbor: tuple[int, int]) -> None:
    set_edge(wall_map, cell[0], cell[1], _wall_between(cell, neighbor), 1)


def interior_edges(rows: int, cols: int) -> list[tuple[int, int, int]]:
    """All interior edges, each once, as ``(r, c, wall)`` with wall DOWN or RIGHT."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                edges.append((r, c, WALL_RIGHT))
            if r < rows - 1:
                edges.append((r, c, WALL_DOWN))
    return edges


def neighbors(wall_map: np.ndarray, r: int, c: int) -> Iterable[tuple[int, int]]:
    """4-neighbours of (r, c) reachable through an open edge."""
    rows, cols = wall_map.shape[:2]
    for dr, dc, wall, _ in _DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and wall_map[r, c, wall] == 1:
            yield nr, nc


def reachable_cells(wall_map: np.ndarray, start: tuple[int, int] = (0, 0)) -> set[tuple[int, int]]:
    queue = deque([start])
    visited = {start}
    while queue:
        r, c = queue.popleft()
        for nb in neighbors(wall_map, r, c):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def shortest_path_steps(wall_map: np.ndarray, start: tuple[int, int], goals: Iterable[tuple[int, int]]) -> int:
    """BFS distance (4-connected moves) from ``start`` to the nearest goal; -1 if none."""
    goals = set(map(tuple, goals))
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        cell, steps = queue.popleft()
        if cell in goals:
            return steps
        for nb in neighbors(wall_map, *cell):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, steps + 1))
    return -1


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def wall_map_problems(wall_map) -> list[str]:
    """Return a list of human-readable problems; empty means the map is valid."""
    if not isinstance(wall_map, np.ndarray):
        return ["wall_map is not a numpy array"]
    if wall_map.ndim != 3 or wall_map.shape[2] != 4 or wall_map.shape[0] < 1 or wall_map.shape[1] < 1:
        return [f"bad shape {wall_map.shape}, expected (rows, cols, 4)"]
    problems = []
    if not np.all((wall_map == 0) | (wall_map == 1)):
        problems.append("values other than 0/1")
    if not (
        np.all(wall_map[0, :, WALL_UP] == 0)
        and np.all(wall_map[-1, :, WALL_DOWN] == 0)
        and np.all(wall_map[:, 0, WALL_LEFT] == 0)
        and np.all(wall_map[:, -1, WALL_RIGHT] == 0)
    ):
        problems.append("outer border is not closed")
    if not np.array_equal(wall_map[1:, :, WALL_UP], wall_map[:-1, :, WALL_DOWN]):
        problems.append("UP/DOWN sides of a horizontal edge disagree")
    if not np.array_equal(wall_map[:, 1:, WALL_LEFT], wall_map[:, :-1, WALL_RIGHT]):
        problems.append("LEFT/RIGHT sides of a vertical edge disagree")
    rows, cols = wall_map.shape[:2]
    n_reach = len(reachable_cells(wall_map, (0, 0)))
    if n_reach != rows * cols:
        problems.append(f"not connected: {n_reach}/{rows * cols} cells reachable from (0, 0)")
    return problems


def validate_wall_map(wall_map) -> bool:
    """Borders closed, edge symmetry, full connectivity. Any size."""
    return not wall_map_problems(wall_map)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def generate_maze(
    rows: int,
    cols: int,
    seed: int,
    loop_fraction: float = 0.0,
    start: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Random maze: DFS recursive backtracker, then random loop injection.

    The DFS part is ``map_generator._dense_photo_maze`` with ``rows, cols`` as
    parameters (same RNG calls, so on 15x20 with the same seed and start it
    yields the same perfect maze). It produces a spanning tree: exactly one
    route between any two cells.

    Loop injection replaces Project A's hand-picked ``loop_edges``: of the
    interior walls still closed after the DFS, ``round(loop_fraction * n)`` are
    chosen uniformly at random (same RNG) and opened. ``loop_fraction=0`` gives
    a perfect maze, ``1`` an empty box.
    """
    if not 0.0 <= loop_fraction <= 1.0:
        raise ValueError("loop_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    wall_map = closed_wall_map(rows, cols)
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack[-1]
        candidates = []
        for dr, dc, _wall, _opp in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                candidates.append((nr, nc))
        if not candidates:
            stack.pop()
            continue
        nr, nc = candidates[int(rng.integers(len(candidates)))]
        open_between(wall_map, (r, c), (nr, nc))
        visited.add((nr, nc))
        stack.append((nr, nc))

    closed = [(r, c, w) for r, c, w in interior_edges(rows, cols) if wall_map[r, c, w] == 0]
    n_open = int(round(loop_fraction * len(closed)))
    if n_open:
        for i in rng.choice(len(closed), size=n_open, replace=False):
            r, c, w = closed[int(i)]
            set_edge(wall_map, r, c, w, 1)
    return wall_map


# --------------------------------------------------------------------------
# Predecessor ".maze" format (Abudi & Huly, micromouse-py-sim, MIT licence)
# --------------------------------------------------------------------------

# Predecessor wall bit flags (sim.maze.Walls).
BIT_N, BIT_E, BIT_S, BIT_W = 0x1, 0x2, 0x4, 0x8
_BIT_TO_CHANNEL = ((BIT_N, WALL_UP), (BIT_S, WALL_DOWN), (BIT_W, WALL_LEFT), (BIT_E, WALL_RIGHT))


def parse_maze_text(text: str, cell_height: int = 1, cell_width: int = 3, empty: str = " ") -> np.ndarray:
    """Parse an ASCII ``.maze`` drawing into a ``(rows, cols)`` array of wall bits.

    Port of ``sim.maze.Maze.from_maze_text``: every cell is a box of
    ``cell_height x cell_width`` characters framed by wall characters; a wall is
    present when the character in the middle of that side is not ``empty``.
    Bits: N=1, E=2, S=4, W=8. Row 0 is the first (top) line.
    """
    lines = text.splitlines(keepends=False)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("maze is empty")
    if not all(len(line) == len(lines[0]) for line in lines):
        raise ValueError("maze is not a rectangle")

    def _exact_div(a: int, b: int) -> int:
        div, mod = divmod(a, b)
        if mod:
            raise ValueError(f"invalid sizes: expected {a} to be divisible by {b}")
        return div

    height = _exact_div(len(lines) - 1, cell_height + 1)
    width = _exact_div(len(lines[0]) - 1, cell_width + 1)
    bits = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        top = row * (cell_height + 1)
        bot = top + cell_height + 1
        mid_r = (top + bot) // 2
        for col in range(width):
            left = col * (cell_width + 1)
            right = left + cell_width + 1
            mid_c = (left + right) // 2
            b = 0
            if lines[top][mid_c] != empty:
                b |= BIT_N
            if lines[mid_r][right] != empty:
                b |= BIT_E
            if lines[bot][mid_c] != empty:
                b |= BIT_S
            if lines[mid_r][left] != empty:
                b |= BIT_W
            bits[row, col] = b
    return bits


def bits_to_wall_map(bits: np.ndarray) -> np.ndarray:
    """Predecessor wall bits (set = wall) -> ``wall_map`` (0 = wall, 1 = open)."""
    bits = np.asarray(bits)
    wall_map = np.ones(bits.shape + (4,), dtype=np.uint8)
    for bit, channel in _BIT_TO_CHANNEL:
        wall_map[..., channel] = np.where(bits & bit, 0, 1)
    return wall_map


def maze_to_wall_map(source: str | Path, cell_height: int = 1, cell_width: int = 3) -> np.ndarray:
    """Convert a predecessor ``.maze`` drawing to a ``wall_map``.

    ``source`` is a path to a ``.maze``/``.txt`` file, or the drawing itself
    (any string containing a newline). The result is not validated; call
    ``validate_wall_map`` on it.
    """
    if isinstance(source, str) and "\n" in source:
        text = source
    else:
        text = Path(source).read_text(encoding="utf-8")
    return bits_to_wall_map(parse_maze_text(text, cell_height, cell_width))


def render_ascii(wall_map: np.ndarray, goal_cells: Iterable[tuple[int, int]] = ()) -> str:
    """Draw a ``wall_map`` in the predecessor ``.maze`` style (``+---+`` / ``|``).

    Goal cells are marked `` G ``. The drawing parses back with
    ``maze_to_wall_map`` to the same ``wall_map``.
    """
    rows, cols = wall_map.shape[:2]
    goals = set(map(tuple, goal_cells))
    out = []
    for r in range(rows):
        top = "+"
        for c in range(cols):
            top += ("---" if wall_map[r, c, WALL_UP] == 0 else "   ") + "+"
        out.append(top)
        mid = "|" if wall_map[r, 0, WALL_LEFT] == 0 else " "
        for c in range(cols):
            mid += " G " if (r, c) in goals else "   "
            mid += "|" if wall_map[r, c, WALL_RIGHT] == 0 else " "
        out.append(mid)
    bot = "+"
    for c in range(cols):
        bot += ("---" if wall_map[rows - 1, c, WALL_DOWN] == 0 else "   ") + "+"
    out.append(bot)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Sensor signatures and ambiguity
# --------------------------------------------------------------------------

def ray_distances(wall_map: np.ndarray) -> np.ndarray:
    """Unbounded per-ray wall distance, shape ``(rows, cols, 4)`` int.

    Channel order is the wall_map order (UP, DOWN, LEFT, RIGHT). Entry ``d`` is
    the number of open cells the ray passes before the first wall, i.e. how
    many cardinal steps the robot could take in that direction. ``d = 0`` means
    the cell's own edge on that side is a wall. Same quantity as
    ``map_generator._wall_distance``.
    """
    rows, cols = wall_map.shape[:2]
    dist = np.zeros((rows, cols, 4), dtype=np.int64)
    # Dynamic programming: d(cell, dir) = 0 if wall else 1 + d(next cell, dir).
    for r in range(rows):  # UP: depends on the row above
        dist[r, :, WALL_UP] = np.where(wall_map[r, :, WALL_UP] == 1, 1 + (dist[r - 1, :, WALL_UP] if r else 0), 0)
    for r in range(rows - 1, -1, -1):  # DOWN
        nxt = dist[r + 1, :, WALL_DOWN] if r < rows - 1 else 0
        dist[r, :, WALL_DOWN] = np.where(wall_map[r, :, WALL_DOWN] == 1, 1 + nxt, 0)
    for c in range(cols):  # LEFT
        prv = dist[:, c - 1, WALL_LEFT] if c else 0
        dist[:, c, WALL_LEFT] = np.where(wall_map[:, c, WALL_LEFT] == 1, 1 + prv, 0)
    for c in range(cols - 1, -1, -1):  # RIGHT
        nxt = dist[:, c + 1, WALL_RIGHT] if c < cols - 1 else 0
        dist[:, c, WALL_RIGHT] = np.where(wall_map[:, c, WALL_RIGHT] == 1, 1 + nxt, 0)
    return dist


def sensor_signature(wall_map: np.ndarray, R: int | None = 2) -> np.ndarray:
    """Noiseless reading of 4 cardinal rays, shape ``(rows, cols, 4)`` int.

    Reading per ray = ``d`` (see ``ray_distances``) if ``d <= R``, else ``NONE``
    (-1). ``R=None`` returns the unbounded distances (Project A signature).
    Channel order UP, DOWN, LEFT, RIGHT. The robot's heading does not enter:
    rays are in absolute (map) directions, matching the 8-absolute-direction
    motion model.
    """
    dist = ray_distances(wall_map)
    if R is None:
        return dist
    if R < 0:
        raise ValueError("R must be >= 0")
    return np.where(dist <= R, dist, NONE)


def signature_groups(signatures: np.ndarray) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    """Group cells by signature: ``{signature tuple: [(r, c), ...]}``."""
    rows, cols = signatures.shape[:2]
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            groups.setdefault(tuple(int(v) for v in signatures[r, c]), []).append((r, c))
    return groups


def group_stats(sizes: Iterable[int]) -> dict:
    """Ambiguity numbers for a partition of the N cells into signature groups.

    - ``n_cells``: N.
    - ``n_shared``: cells whose signature is shared with at least one other cell
      (Project A's ``ambiguity_score``). The headline number.
    - ``n_distinct``: number of distinct signatures.
    - ``amb = 1 - n_distinct / N``: 0 when every cell is unique.
    - ``largest_group``: size of the biggest group of look-alike cells.
    - ``H1``: expected posterior entropy (nats) after one noiseless observation
      from a uniform belief, ``sum_g (|g|/N) ln|g|``. 0 = one look localizes,
      ``ln N`` = the observation tells nothing.
    """
    sizes = [int(s) for s in sizes]
    n = sum(sizes)
    return {
        "n_cells": n,
        "n_shared": sum(s for s in sizes if s > 1),
        "n_distinct": len(sizes),
        "amb": 1.0 - len(sizes) / n,
        "largest_group": max(sizes),
        "H1": sum((s / n) * math.log(s) for s in sizes),
    }


def ambiguity(wall_map: np.ndarray, R: int = 2) -> dict:
    """How many cells share a sensor signature, two ways.

    Returns ``{"R": R, "unbounded": stats, "limited": stats}`` where each
    ``stats`` dict is ``group_stats`` (``n_cells, n_shared, n_distinct, amb,
    largest_group, H1``):

    - ``unbounded``: Project A's 4-direction wall-distance signature
      (``map_generator._wall_distance_signature``), independent of R.
      ``unbounded["n_shared"]`` equals Project A's ``score_map`` ambiguity.
    - ``limited``: the planned sensor, 4 cardinal rays with range R, noiseless
      (``sensor_signature(wall_map, R)``).
    """
    return {
        "R": R,
        "unbounded": group_stats(len(g) for g in signature_groups(sensor_signature(wall_map, None)).values()),
        "limited": group_stats(len(g) for g in signature_groups(sensor_signature(wall_map, R)).values()),
    }


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def _split_name(name: str) -> tuple[str, str]:
    """Split ``"<size>/<name>"``; reject anything else (bare, nested, odd characters)."""
    parts = name.split("/") if isinstance(name, str) else []
    if len(parts) != 2 or not all(_NAME_PART.fullmatch(p) for p in parts):
        raise ValueError(
            f"maze name must look like '<size>/<name>' with each part made of letters, digits, '_' or '-'; got {name!r}"
        )
    return parts[0], parts[1]


def maze_path(name: str, root: Path | None = None) -> Path:
    size, short = _split_name(name)
    return Path(root or MAZE_DIR) / size / f"{short}.npz"


def _read_index(root: Path | None = None) -> dict:
    path = Path(root or MAZE_DIR) / "index.yaml"
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("mazes", {}) or {}


def list_mazes(size: str | None = None, root: Path | None = None) -> list[str]:
    """Names of the stored mazes (``<size>/<name>``), sorted; optionally one size only."""
    base = Path(root or MAZE_DIR)
    names = sorted(
        f"{p.parent.name}/{p.stem}"
        for p in base.glob("*/*.npz")
    )
    if size is not None:
        names = [n for n in names if n.split("/", 1)[0] == size]
    return names


def resolve_name(name: str, root: Path | None = None) -> str:
    """Return the canonical ``<size>/<name>`` for ``name`` or raise ``KeyError``.

    Matching is exact and case-sensitive against ``list_mazes()`` on every OS
    (so ``"MEDIUM/m10_a"`` fails on Windows too). A bare name (``"m10_a"``) is
    accepted when exactly one size has it.
    """
    if not isinstance(name, str):
        raise KeyError(f"maze name must be a string, got {name!r}")
    known = list_mazes(root=root)
    if "/" in name:
        if name not in known:
            raise KeyError(f"maze {name!r} not found; known mazes: {known}")
        return name
    matches = [n for n in known if n.split("/", 1)[1] == name]
    if not matches:
        raise KeyError(f"maze {name!r} not found; known mazes: {known}")
    if len(matches) > 1:
        raise KeyError(f"bare maze name {name!r} is ambiguous, use one of {matches}")
    return matches[0]


def load_maze(name: str, root: Path | None = None) -> MazeSpec:
    """Load a stored maze by name, e.g. ``load_maze("medium/m10_a")``.

    The name must be one of ``list_mazes()`` exactly (case-sensitive). A bare
    name (``"m10_a"``) is accepted when it is unique across sizes. Raises
    ``KeyError`` for an unknown or ambiguous name.
    ``meta`` is the maze's entry in ``index.yaml`` (empty if not indexed).
    Each call returns a fresh, writable copy of the wall_map.
    """
    name = resolve_name(name, root)
    path = maze_path(name, root)
    with np.load(path, allow_pickle=False) as data:
        wall_map = np.array(data["wall_map"], dtype=np.uint8)
        goals = tuple((int(r), int(c)) for r, c in np.asarray(data["goal_cells"]).reshape(-1, 2))
        stored_meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    meta = dict(stored_meta)
    meta.update(_read_index(root).get(name, {}))
    rows, cols = wall_map.shape[:2]
    return MazeSpec(name=name, wall_map=wall_map, goal_cells=goals, rows=rows, cols=cols, meta=meta)


def save_maze(spec: MazeSpec, root: Path | None = None, write_ascii: bool = True, check: bool = True) -> Path:
    """Write ``<root>/<size>/<name>.npz`` (+ ``.txt`` drawing). Returns the npz path.

    Always enforced (raises ``ValueError``): the name is ``<size>/<name>`` (no
    nesting), ``wall_map`` is ``(rows, cols, 4)``, the shape matches the size
    class for ``small``/``medium``/``large``, and ``goal_cells`` is non-empty
    with every cell inside the map. With ``check=True`` (default) the wall_map
    must also pass ``validate_wall_map``.
    Does not touch ``index.yaml``; ``build_maze_bank.py`` rewrites that.
    """
    size, _short = _split_name(spec.name)
    shape = np.shape(spec.wall_map)
    if len(shape) != 3 or shape[2] != 4 or tuple(shape[:2]) != (spec.rows, spec.cols):
        raise ValueError(f"maze {spec.name}: wall_map shape {shape} does not match rows x cols = {spec.rows}x{spec.cols}")
    if size in SIZE_SHAPES and (spec.rows, spec.cols) != SIZE_SHAPES[size]:
        raise ValueError(f"maze {spec.name}: {size} mazes must be {SIZE_SHAPES[size]}, got {(spec.rows, spec.cols)}")
    if not spec.goal_cells:
        raise ValueError(f"maze {spec.name}: goal_cells is empty")
    for goal in spec.goal_cells:
        if len(goal) != 2 or not (0 <= goal[0] < spec.rows and 0 <= goal[1] < spec.cols):
            raise ValueError(f"maze {spec.name}: goal {tuple(goal)} outside the {spec.rows}x{spec.cols} map")
    if check:
        problems = wall_map_problems(spec.wall_map)
        if problems:
            raise ValueError(f"refusing to save invalid maze {spec.name}: {problems}")
    path = maze_path(spec.name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wall_map=np.asarray(spec.wall_map, dtype=np.uint8),
        goal_cells=np.asarray(spec.goal_cells, dtype=np.int64).reshape(-1, 2),
        meta_json=np.array(json.dumps(spec.meta, sort_keys=True, default=_json_default)),
    )
    if write_ascii:
        path.with_suffix(".txt").write_text(render_ascii(spec.wall_map, spec.goal_cells), encoding="utf-8")
    return path


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (tuple, set)):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")
