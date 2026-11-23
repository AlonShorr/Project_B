"""Build a MuJoCo MJCF model of a maze plus the setup guide's point robot (m1.0, m1.1).

The maze part (``maze_worldbody``) is shared with the mecanum robot (mecanum.py).

Geometry: 180 mm cell pitch and 10 mm x 50 mm walls centred on cell boundaries
(the mecanum brief's real maze; the setup guide used IEEE's 12 mm).
**There are no corner posts** (team decision, 23.11.2026): the real maze has
none, and the Python model (motion_model.can_step, D2) lets a diagonal pass a
corner when all four edges that meet there are open. Each wall is one cell
plus one wall thickness long, so walls that meet at a corner overlap and close
it, and a corner with no wall at all stays empty. A centre-to-centre diagonal
through an empty corner clears every wall by more than 60 mm.

World frame: x right, y UP (grid row r maps to y = -(r + 0.5) * CELL), z up.

``wall_map`` is the Project B format (``lidar_sim``): shape ``(rows, cols, 4)``,
channels ``config.WALL_UP/DOWN/LEFT/RIGHT``, ``0 = wall`` and ``1 = open``.
This differs from the setup guide, which assumed a 2D bitmask with set = wall.
"""

from __future__ import annotations

import numpy as np

from . import LIDAR_SIM_DIR  # noqa: F401  (puts lidar_sim on sys.path)
from config import CELL_SIZE_MM, WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from maze_bank import wall_map_problems

CELL = CELL_SIZE_MM / 1000.0   # m (0.180)
WALL_T = 0.010                 # m (mecanum brief)
WALL_H = 0.050                 # m
ROBOT_R = 0.050                # m (100 mm footprint)
ROBOT_HALF_H = 0.015
RAY_Z = 0.020                  # m above floor; must be < WALL_H or every ray misses

# Distance from the cell centre to the inner face of a wall on its edge. The
# rangefinder sites sit at the robot centre, so a wall at ray distance j
# (sensing.py: j = 0 is the robot's own edge) reads WALL_OFFSET + j * CELL.
WALL_OFFSET = CELL / 2 - WALL_T / 2   # 0.085 m

# Reflectance per geom kind, read by the mecanum ToF cone model (assumed values).
REFLECTANCE = {"wall": 0.5, "floor": 0.3}

# VL53L4CD: 1 to 1,300 mm. The rangefinder cutoff is the sensor's own range, not
# R: the sensor always returns a distance, and "no wall within R" is decided when
# the reading is discretized (env.discretize_ranges), as it would be on the robot.
SENSOR_MIN_RANGE = 0.001       # m
SENSOR_MAX_RANGE = 1.300       # m


def max_sensing_R() -> int:
    """Largest R whose 'wall at R' vs 'wall at R + 1' boundary lies inside the sensor range."""
    return int(np.floor((SENSOR_MAX_RANGE - WALL_OFFSET) / CELL - 0.5))   # 6


# Rays in sensing.py order (N, E, S, W), as world (x, y) directions.
RAY_NAMES = ("N", "E", "S", "W")
RAY_DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def cell_center_world(row: int, col: int) -> tuple[float, float]:
    """Grid cell -> MuJoCo (x, y) of its centre. Grid y is down, world y is up."""
    return (col + 0.5) * CELL, -(row + 0.5) * CELL


def _is_wall(wall_map: np.ndarray, r: int, c: int, channel: int) -> bool:
    return wall_map[r, c, channel] == 0


def _box(name: str, cx: float, cy: float, sx: float, sy: float) -> str:
    """One wall box; (cx, cy) in grid coords (y down), sizes are full lengths."""
    return (f'<geom name="{name}" class="wall" pos="{cx:.4f} {-cy:.4f} {WALL_H / 2:.4f}" '
            f'size="{sx / 2:.4f} {sy / 2:.4f} {WALL_H / 2:.4f}"/>')


def wall_geoms(wall_map: np.ndarray) -> list[str]:
    """Emit each wall exactly once: UP and LEFT of every cell, plus the bottom and
    right outer border. No posts. Raises ValueError if the two sides of an edge
    disagree (then 'emit once' would be ambiguous)."""
    wall_map = np.asarray(wall_map)
    problems = wall_map_problems(wall_map)
    if problems:
        raise ValueError("inconsistent wall_map: " + "; ".join(problems))
    rows, cols = wall_map.shape[:2]
    seg = CELL + WALL_T                # overlaps the neighbouring wall by half a thickness at each end
    g: list[str] = []
    for r in range(rows):
        for c in range(cols):
            if _is_wall(wall_map, r, c, WALL_UP):
                g.append(_box(f"wall_U_{r}_{c}", (c + .5) * CELL, r * CELL, seg, WALL_T))
            if _is_wall(wall_map, r, c, WALL_LEFT):
                g.append(_box(f"wall_L_{r}_{c}", c * CELL, (r + .5) * CELL, WALL_T, seg))
    for c in range(cols):
        if _is_wall(wall_map, rows - 1, c, WALL_DOWN):
            g.append(_box(f"wall_B_{c}", (c + .5) * CELL, rows * CELL, seg, WALL_T))
    for r in range(rows):
        if _is_wall(wall_map, r, cols - 1, WALL_RIGHT):
            g.append(_box(f"wall_R_{r}", cols * CELL, (r + .5) * CELL, WALL_T, seg))
    return g


def maze_worldbody(wall_map: np.ndarray) -> str:
    """Light, floor and walls (all in geom group 0, which the ToF rays see)."""
    wall_map = np.asarray(wall_map)
    rows, cols = wall_map.shape[:2]
    w, h = cols * CELL, rows * CELL
    walls = "\n    ".join(wall_geoms(wall_map))
    return f"""<light pos="{w / 2:.3f} {-h / 2:.3f} 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="{w / 2:.3f} {-h / 2:.3f} 0" size="{w:.3f} {h:.3f} .01"
          rgba=".15 .15 .15 1" contype="0" conaffinity="0" group="0"/>
    {walls}"""


WALL_DEFAULT = '<default class="wall"><geom type="box" rgba=".85 .85 .85 1" contype="1" conaffinity="1" group="0"/></default>'


def build_mjcf(wall_map: np.ndarray, start: tuple[int, int], R: int,
               timestep: float = 0.002, kv: float = 5.0, force_limit: float = 2.0) -> str:
    """Return the full MJCF string. ``start`` = (row, col) of the robot's cell.

    Four rangefinders in N, E, S, W order with the sensor's own range as cutoff.
    8 rays are not supported: sensing.py has 4 (see its module docstring).
    """
    if not 0 <= R <= max_sensing_R():
        raise ValueError(f"R = {R} needs walls beyond the {SENSOR_MAX_RANGE} m sensor range "
                         f"(max R = {max_sensing_R()})")
    wall_map = np.asarray(wall_map)
    rows, cols = wall_map.shape[:2]
    x0, y0 = cell_center_world(*start)
    sites = "\n      ".join(
        f'<site name="ray_{k}" pos="0 0 0" zaxis="{RAY_DIRS[k][0]} {RAY_DIRS[k][1]} 0" size="0.004"/>'
        for k in RAY_NAMES)
    sensors = "\n    ".join(
        f'<rangefinder name="rf_{k}" site="ray_{k}" cutoff="{SENSOR_MAX_RANGE:.4f}"/>' for k in RAY_NAMES)
    return f"""<mujoco model="maze_{rows}x{cols}">
  <option timestep="{timestep}"/>
  <default>
    {WALL_DEFAULT}
  </default>
  <worldbody>
    {maze_worldbody(wall_map)}
    <body name="robot" pos="{x0:.4f} {y0:.4f} {RAY_Z:.4f}">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <geom name="robot_geom" type="cylinder" size="{ROBOT_R} {ROBOT_HALF_H}" mass="0.25"
            rgba=".2 .55 1 1" contype="1" conaffinity="1"/>
      {sites}
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="slide_x" kv="{kv}" forcelimited="true" forcerange="-{force_limit} {force_limit}"/>
    <velocity name="vy" joint="slide_y" kv="{kv}" forcelimited="true" forcerange="-{force_limit} {force_limit}"/>
  </actuator>
  <sensor>
    {sensors}
  </sensor>
</mujoco>
"""
