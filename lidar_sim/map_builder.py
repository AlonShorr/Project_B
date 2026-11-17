import numpy as np

from config import (
    MAP_ROWS, MAP_COLS, GRID_SIZE, MAP_WIDTH, MAP_HEIGHT,
    WALL_UP, WALL_DOWN, WALL_LEFT, WALL_RIGHT,
)
from map_generator import get_map


def generate_wall_map():
    """
    Return the default generated map using thin walls.
    Returns an ndarray of shape (Rows, Cols, 4). Values: 0=Wall, 1=Open.
    """
    return np.array(get_map(0)["wall_map"], copy=True)


def toggle_wall_click(wall_map, mx, my):
    """
    Toggle the wall edge nearest to mouse position (mx, my) in the map area.
    Keeps both sides of the shared edge in sync.
    """
    if not (0 <= mx < MAP_WIDTH and 0 <= my < MAP_HEIGHT):
        return

    c = mx // GRID_SIZE
    r = my // GRID_SIZE
    lx = mx % GRID_SIZE
    ly = my % GRID_SIZE

    d_left = lx
    d_right = GRID_SIZE - lx
    d_up = ly
    d_down = GRID_SIZE - ly
    min_dist = min(d_left, d_right, d_up, d_down)

    CLICK_THRESHOLD = 15
    if min_dist > CLICK_THRESHOLD:
        return

    if min_dist == d_up:
        val = 1 - wall_map[r, c, WALL_UP]
        wall_map[r, c, WALL_UP] = val
        if r > 0:
            wall_map[r - 1, c, WALL_DOWN] = val
    elif min_dist == d_down:
        val = 1 - wall_map[r, c, WALL_DOWN]
        wall_map[r, c, WALL_DOWN] = val
        if r < MAP_ROWS - 1:
            wall_map[r + 1, c, WALL_UP] = val
    elif min_dist == d_left:
        val = 1 - wall_map[r, c, WALL_LEFT]
        wall_map[r, c, WALL_LEFT] = val
        if c > 0:
            wall_map[r, c - 1, WALL_RIGHT] = val
    elif min_dist == d_right:
        val = 1 - wall_map[r, c, WALL_RIGHT]
        wall_map[r, c, WALL_RIGHT] = val
        if c < MAP_COLS - 1:
            wall_map[r, c + 1, WALL_LEFT] = val


def dump_selected_cells(data, selected_cells):
    """Print expected sensor readings for every selected cell to the console."""
    print("\n=== SELECTED CELLS DUMP (PYTHON) ===")
    for (py_r, c) in sorted(selected_cells):
        print(f"Cell ({py_r}, {c}):")
        for a in range(8):
            vals = ", ".join([f"{x:6.2f}" for x in data[py_r, c, a] / 1000])
            print(f"  Angle {a} ({a * 45:3d}°): {{ {vals} }},")
    print("=== END DUMP ===\n")
