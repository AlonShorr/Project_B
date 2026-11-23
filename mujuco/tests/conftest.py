"""Make ``sim`` importable when pytest runs from the repo root or from mujuco/."""

import sys
from pathlib import Path

MUJUCO_DIR = Path(__file__).resolve().parents[1]
if str(MUJUCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJUCO_DIR))

import sim  # noqa: E402,F401  (adds lidar_sim to sys.path)
