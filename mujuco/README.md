# MuJoCo simulator (phase pm)

The code from the *MuJoCo Setup Guide — Kidnapped Robot Project (Phase pm)*, adapted to the conventions of `lidar_sim/` and to the team's decisions. All MuJoCo code lives here. `lidar_sim/` never imports `mujoco`.

**Status:** m1.0 to m1.4 are done (numbers below). The m2 sources and the m3 harness are wired in, but their parameters are placeholders until m2.4.

## Install

Use the project's existing `.venv` (Python 3.13) at the repo root. Don't create a new one: on this machine `python3` is 3.15 and plain `python` is 3.7.

```powershell
.venv\Scripts\python -m pip install -r mujuco\requirements-sim.txt
.venv\Scripts\python -c "import mujoco; print(mujoco.__version__)"   # 4.0.1
```

`requirements-sim.txt` is `pip freeze` of the whole venv, so it also pins the `lidar_sim` packages.

## Layout

```
mujuco/
  configs/sim_default.yaml  controller gains + m2 noise flags and parameters (all noise off = m1)
  sim/
    __init__.py       puts lidar_sim/ on sys.path; BUILD_DIR
    mjcf.py           wall_map -> MJCF string, no corner posts (m1.0, m1.1)
    env.py            MazeSim: step(d, k) / step_action(Action, k), observe(), observe_index() (m1.2)
    noise.py          NoiseConfig + the m2 sources
    noisy_env.py      NoisyMazeSim: closes the loop on odometry (m2)
    mj_grid_env.py    MujocoGridEnv: GridEnv's interface on MuJoCo (m1.3)
    mj_config.py      loads configs/sim_default.yaml with --mj-set overrides
    calibrate.py      m3 batch runs, Wilson / Dirichlet fits, YAML export
  tests/              pytest (60 tests)
  run_mujoco.py       lidar_sim's runner on MuJoCo: same flags, same CSV (m1.3, m1.4)
  smoke_test.py       guide §5
  run_calibration.py  guide §7
  build/              generated .xml and placeholder calibration output (git-ignored)
```

## Running

Run from `mujuco/`:

```powershell
..\.venv\Scripts\python -m pytest tests -q
..\.venv\Scripts\python smoke_test.py
..\.venv\Scripts\python run_mujoco.py --policy projecta --set maze.name=small/m5_a --n-runs 100
..\.venv\Scripts\python run_mujoco.py --policy projecta --mj-set noise.bad_return_on=true --mj-set noise.p_bad_return=0.1
..\.venv\Scripts\python run_calibration.py            # writes build\calibrated.yaml
..\.venv\Scripts\python -m mujoco.viewer --mjcf=build\maze_smoke.xml
```

`run_mujoco.py` takes every `runner.py` flag (`--config`, `--set`, `--policy`, `--n-runs`, `--workers`, ...) plus `--mj-config` and `--mj-set`. It writes the same CSV columns as the Python simulator, to `results/mujoco_<policy>.csv` by default, plus two sidecars: `<out>.config.yaml` (the experiment) and `<out>.mujoco.yaml` (controller, noise, mujoco version).

Expected smoke-test output (rays in N, E, S, W order; wall-clock times vary):

```
mujoco 4.0.1 | geoms: 19 | wall/post geoms: 17
start (1, 1) ranges N,E,S,W (m): [0.264 0.084 0.444 0.264] obs: [1 0 2 1]
E x1 (blocked)  -> cell (1, 1) collided=True sim 0.30s ...
S x2            -> cell (3, 1) collided=False sim 2.13s ...
E x2            -> cell (3, 3) collided=False sim 2.13s ...
NW x1           -> cell (2, 2) collided=False sim 1.59s ...
```

## Team decisions (23.11.2026)

1. **No corner posts.** The real maze has none, so the MJCF has only the walls in `wall_map`. Each wall is one cell plus one wall thickness long: walls that meet at a corner overlap and close it, and a corner with no wall stays empty. A diagonal therefore works exactly as `motion_model.can_step` (D2) says: it passes when all four edges at its corner are open. `test_every_move_matches_the_D2_rule` checks all 8 moves from every cell of every 5×5 maze.
2. **The sensor never returns "nothing".**
   - A ray with no wall within the VL53L4CD's 1.3 m reads 1.3 m. The rangefinder cutoff is the sensor range, not R, so the earlier cutoff problem is gone.
   - "No wall within R" (digit R + 1) is decided when the reading is discretized. That limits R to 6.
   - **A miss is a ray that doesn't return properly.** With probability `p_bad_return` the reading gets extra noise N(0, `bad_return_sigma`). It discretizes to a wrong wall distance (closer or farther), never to an empty reading.
   - `calibrate.p_miss` reports the per-ray rate of wrong digits plus the full confusion matrix `confusion[true][observed]`.

## m1 results (m1.4)

100 headless episodes, `projecta`, 1 worker, default config, noise off:

| Maze | Time | Outcomes |
|---|---|---|
| small/m5_a (5×5) | 2.1 s | 100 success |
| medium/m10_a (10×10), 4 workers | 2.0 s | 100 success |

One action takes about 10 ms of wall time. The target was 10 minutes.

## Changes from the guide

1. **`wall_map` format.** The repo uses `(rows, cols, 4)` with channels `config.WALL_UP/DOWN/LEFT/RIGHT`, and **0 = wall, 1 = open**. The guide assumed a 2D bitmask. Maps whose two sides of an edge disagree are rejected.
2. **Observations follow `sensing.py`.**
   - The rays are N, E, S, W.
   - Each ray gives a digit 0..R, or R + 1.
   - `observe_index()` packs the digits with `sensing.encode_observation`.
   - Every cell of every bank maze gives exactly `sensing.noiseless_signatures`.
3. **`R` defaults to `config.SENSING_RANGE_R`** (2). Only 4 rays, because `sensing.py` has 4.
4. **`dirichlet_summary`:** a context with only one possible outcome (a blocked move) gave a `nan` interval. It now returns p = 1 with [1, 1].
5. **Calibration contexts:** diagonals forbidden by D2 are `blocked` and kept out of `free`, like blocked cardinal moves.
6. **Calibration output** goes to `build/` by default. Write to `lidar_sim/configs/calibrated.yaml` (`--out`) only with measured m2.4 values.

## Known issues for m2 (placeholder noise)

- **The friction model makes collisions.** The guide's odometry integrates the *commanded* velocity. With friction on, the robot stalls about 40 mm short of each cell centre and doesn't notice. The offset adds up until the robot scrapes a corridor wall (34 mm clearance each side). With friction alone, `projecta` on 5×5 collides in 57% of episodes. Command noise alone gives 4%, and stop jitter alone 0%. A real robot's encoders see that its wheels didn't turn, so for m2.4 odometry should integrate the measured wheel motion (with slip noise). Adding a correction that recentres the robot from its side walls is another option.
- **`sensing.py` doesn't model noisy readings yet.** D6 lets a miss only skip a wall (the reading moves farther). Under decision 2 a miss can also read closer, so the belief filter sees "impossible" observations and falls back to mixing with the uniform belief. With bad returns on, that happened in 25 of 100 episodes. The fix is in `sensing.py` (t4): its observation model should match decision 2, with its numbers fitted to the confusion matrix from `calibrate.p_miss`.

## Not done yet

- `render.py` (m4.3 videos).
