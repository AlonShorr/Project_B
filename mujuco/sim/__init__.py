"""MuJoCo simulator for Project B (phase pm: m1 to m3).

All MuJoCo code lives in this package. The Project B core in ``lidar_sim/``
never imports ``mujoco``. Data flows one way: ``sim`` reads ``wall_map`` and
the sensing conventions from ``lidar_sim``, and m3 writes a calibrated YAML
that the core's ``Model`` can read.

``lidar_sim`` is a flat folder of modules, not a package, so it is appended
to ``sys.path`` here (the same modules are then importable as ``maze_bank``,
``sensing``, ``config``, ...).
"""

from __future__ import annotations

import sys
from pathlib import Path

LIDAR_SIM_DIR = Path(__file__).resolve().parents[2] / "lidar_sim"
BUILD_DIR = Path(__file__).resolve().parents[1] / "build"

if str(LIDAR_SIM_DIR) not in sys.path:
    sys.path.append(str(LIDAR_SIM_DIR))
