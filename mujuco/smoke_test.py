"""m1 smoke test (setup guide §5). Run from the mujuco folder:

    ..\\.venv\\Scripts\\python smoke_test.py

If the output matches the expected output in README.md, MuJoCo, the MJCF
builder, the controller, the contacts and the rangefinders all work.
"""

import time

import mujoco
import numpy as np

from sim import BUILD_DIR
from sim.env import MazeSim
from config import WALL_RIGHT
from maze_bank import open_wall_map, set_edge
from motion_model import Action

# 4x4 test maze: outer box + an inner wall between (1,1) and (1,2)
wm = open_wall_map(4, 4)
set_edge(wm, 1, 1, WALL_RIGHT, 0)

BUILD_DIR.mkdir(exist_ok=True)
sim = MazeSim(wm, start=(1, 1), R=3, xml_path=str(BUILD_DIR / "maze_smoke.xml"))
print("mujoco", mujoco.__version__, "| geoms:", sim.model.ngeom, "| wall/post geoms:", int(sim.is_wall.sum()))
print("start", sim.cell(), "ranges N,E,S,W (m):", np.round(sim.ranges(), 3), "obs:", sim.observe())

for name, a, k in [("E x1 (blocked)", Action.E, 1), ("S x2", Action.S, 2), ("E x2", Action.E, 2), ("NW x1", Action.NW, 1)]:
    t = time.perf_counter()
    res = sim.step_action(a, k)
    print(f"{name:15s} -> cell {res.cell} collided={res.collided} sim {res.sim_time:.2f}s "
          f"wall {1e3 * (time.perf_counter() - t):.0f} ms | obs {sim.observe()}")
