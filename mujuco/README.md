# MuJoCo simulator (phase pm)

Two robots behind one interface:

- **`mecanum`** (the default) is Level A of the mecanum design proposal (Gantt site, Segev, 23.11.2026). It has four N20-driven mecanum wheels, encoder odometry and VL53L1X time-of-flight cones.
- **`point`** is the setup guide's prototype, a cylinder on slide joints. It stays for comparison.

All MuJoCo code lives here. `lidar_sim/` never imports `mujoco`.

**Status:** m1.0 to m1.4 are done for both robots (numbers below). The noise sources and the m3 calibration harness are wired in, but every noise value is a placeholder until it's calibrated.

## Install

Use the project's existing `.venv` (Python 3.13) at the repo root. Don't create a new one: on this machine `python3` is 3.15 and plain `python` is 3.7.

```powershell
.venv\Scripts\python -m pip install -r mujuco\requirements-sim.txt
.venv\Scripts\python -c "import mujoco; print(mujoco.__version__)"   # 4.0.1
```

## Layout

```
mujuco/
  configs/sim_default.yaml  robot choice, mecanum controller / noise / ToF, point-robot controller / noise
  sim/
    __init__.py       puts lidar_sim/ on sys.path; BUILD_DIR
    mjcf.py           maze walls (10 mm, no posts) shared by both robots, plus the point robot
    mecanum.py        mecanum Level A: RobotConfig, IK/FK, MJCF, controller, MecanumSim
    tof.py            VL53L1X cone model (mj_multiRay, return weighting, noise)
    env.py            point robot MazeSim (m1.2) and the shared discretize_ranges
    noise.py          point robot NoiseConfig
    noisy_env.py      point robot with noise (odometry on commanded velocity)
    mj_grid_env.py    MujocoGridEnv: GridEnv's interface on either robot (m1.3)
    mj_config.py      loads configs/sim_default.yaml with --mj-set overrides
    calibrate.py      m3 batch runs, Wilson / Dirichlet fits, YAML export
  tests/              pytest (about 2 minutes)
  viewer.py           drive the robot from the keyboard, or watch a policy, in the MuJoCo viewer
  run_mujoco.py       lidar_sim's runner on MuJoCo: same flags, same CSV (m1.3, m1.4)
  smoke_test.py       both robots, a short scripted sequence
  run_calibration.py  m3 (--robot mecanum | point)
  build/              generated .xml and placeholder calibration output (git-ignored)
```

## Running

Run from `mujuco/`:

```powershell
..\.venv\Scripts\python -m pytest tests -q
..\.venv\Scripts\python smoke_test.py
..\.venv\Scripts\python run_mujoco.py --policy projecta --set maze.name=small/m5_a --n-runs 100
..\.venv\Scripts\python run_mujoco.py --policy projecta --mj-set robot=point
..\.venv\Scripts\python run_mujoco.py --policy projecta --mj-set mecanum.noise.slip_on=true --mj-set mecanum.tof.fov_deg=27
..\.venv\Scripts\python run_calibration.py --robot mecanum         # writes build\calibrated.yaml
..\.venv\Scripts\python -m mujoco.viewer --mjcf=build\mecanum_smoke.xml
```

## Viewer

```powershell
..\.venv\Scripts\python viewer.py                                   # drive it yourself, random kidnap on small/m5_a
..\.venv\Scripts\python viewer.py --maze medium/m10_a --start 0,0
..\.venv\Scripts\python viewer.py --policy projecta --seed 3        # watch the algorithm
..\.venv\Scripts\python viewer.py --policy projecta --episodes 10 --speed 3
..\.venv\Scripts\python viewer.py --mj-set mecanum.controller.yaw_mode=held --mj-set mecanum.noise.imu_on=true
```

| Key | Action |
|---|---|
| Arrows, or numpad 8 6 2 4 | Move N, E, S, W one cell |
| Home, PgUp, End, PgDn, or numpad 7 9 1 3 | Move NW, NE, SW, SE |
| Enter, or numpad 5 | DECLARE "arrived" |
| Insert | New episode (next seed, random kidnap) |
| Delete | Show or hide the belief heat map |
| Numpad + / − | Faster or slower (policy mode) |
| Pause, or numpad 0 | Pause or resume the policy |

- **Same path as an experiment:** every action goes through `MujocoGridEnv`, the env the batch runner uses, so collisions, wrong declarations and observations are exactly what an experiment records.
- **On the floor:** goal cells are green, and the robot's belief is an orange heat map. The belief is the policy's own in `--policy` mode, and an exact Bayes filter over your moves in manual mode.
- **Text overlay:** the true cell, the observation digits, the belief peak, P(goal) and the outcome.
- **Mouse:** the viewer's own mouse controls (rotate, pan, zoom) still work.

`run_mujoco.py` takes every `runner.py` flag (`--config`, `--set`, `--policy`, `--n-runs`, `--workers`, ...) plus `--mj-config` and `--mj-set key.path=value`. It writes the same CSV columns as the Python simulator, to `results/mujoco_<policy>.csv` by default, plus two sidecars: `<out>.config.yaml` (the experiment) and `<out>.mujoco.yaml` (robot, controller, noise, mujoco version).

Expected smoke-test output (wall-clock times vary):

```
mujoco 4.0.1 | geoms: 19 | wall/post geoms: 17
start (1, 1) ranges N,E,S,W (m): [0.265 0.085 0.445 0.265] obs: [1 0 2 1]
E x1 (blocked)  -> cell (1, 1) collided=True ...
S x2            -> cell (3, 1) collided=False ...
E x2            -> cell (3, 3) collided=False ...
NW x1           -> cell (2, 2) collided=False ...

mecanum Level A | geoms: 25 | mass: 0.3 kg
start (1, 1) ToF N,E,S,W (m): [0.231 0.05  0.407 0.231] obs: [1 0 2 1]
S x2            -> cell (3, 1) collided=False sim 1.52s ...
E x2            -> cell (3, 3) collided=False sim 1.52s ...
NW x1           -> cell (2, 2) collided=False sim 1.17s ...
N x2            -> cell (0, 2) collided=False sim 1.52s ...
N x1 (blocked)  -> cell (0, 2) collided=True  sim 0.92s ...
```

## The mecanum robot (Level A)

What is modelled, from the proposal (§4, §5, §7):

- **Chassis:** 110 × 110 × 40 mm, 300 g (chassis and electronics 180 g, 4 motors at 9.5 g each, battery 50 g, 4 wheels at 8 g each). It slides in x and y with yaw locked, and only its box collides with walls.
- **Wheels and motors:** four 38 mm wheels in X configuration, each on a hinge driven by an N20 50:1 model: torque = 0.084 N·m · duty − (0.084 / 61.8 rad/s) · ω.
- **Level A coupling:** every 1 ms step, the actual wheel speeds go through mecanum FK to a body twist, which force-limited planar servos track (2.35 N, the traction limit). A wall pushes back, so a blocked robot stops while its wheels keep turning.
- **Encoders:** 617.4 counts per wheel turn. Odometry integrates FK of count differences. The controller never sees the true position.
- **Yaw (D1):** `yaw_mode: locked` (default) keeps the heading perfect. `held` adds a yaw hinge and an IMU: a MuJoCo gyro and accelerometer, with white noise and a random-walk gyro bias when `imu_on`. It also adds a yaw-rate disturbance (`yaw_drift_on`), and a heading-hold loop on the integrated gyro rotates the odometry.
- **Controller (every 1 ms physics step):**
  1. A trapezoidal profile (0.30 m/s, 1.0 m/s²) from the believed cell centre to the target centre.
  2. A position loop on odometry. With **strafe compensation** (on by default), the firmware scales the sideways part of its odometry by the robot's calibrated *mean* strafe efficiency. It doesn't see each episode's own draw.
  3. **Wall centering** (on by default): at the 30 Hz ToF rate, a side sensor that sees a wall in the robot's own cell corrects the lateral odometry toward 50 mm from that wall. It only acts within ±60 mm of a cell centre, where the side cone lies fully on the cell's own wall.
  4. **Front-wall alignment** (on by default): in the last 60 mm of a move, the sensor facing a wall on the far edge of the target cell corrects the distance along the move.
  5. Heading hold (held mode only).
  6. IK, then a per-wheel PI with feedforward on encoder speed, giving a PWM duty for the motor model.
- **ToF:** four VL53L1X, 35 mm from the centre and 25 mm up. Each casts 21 rays with return weighting `b(α)·ρ·cos θ / d²`. A reading is 50 mm at a cell centre facing an adjacent wall. The digit is `round((d − 50 mm) / 180 mm)`, and anything past R is "no wall within R".
- **Noise:** all placeholders and all off by default.
  - Motion: per-wheel radius error and strafe efficiency (both per episode), and per-action slip.
  - ToF: noise σ = 3 mm + 1% · d, widened by the cone's return spread.
  - Bad returns (decision 2).

Proposal decisions, as implemented:

| # | Status |
|---|---|
| D1 yaw | **Both modes, locked by default** (team, 23.11). `held` has a gyro heading hold, IMU noise and yaw drift. |
| D2 `heading_k` | **Always faces north** (team, 23.11). Actions and ToF labels are in the maze frame, and the POMDP is unchanged. |
| Wall centering | **On by default** (team, 23.11). m3 must calibrate with the same setting the real robot uses. |
| Strafe compensation, front-wall alignment | **Both on by default** (team, 23.11). The same calibration note applies. |
| Level B | **Later** (team, 23.11), once slip parameters are needed. |
| D3 posts | Settled by team decision 1: no posts. A straight diagonal is legal exactly when D2's four edges are open, and the L-shaped expansion isn't needed. |
| D4 zig-zag test | Doesn't apply without posts. A zig-zag is D2-illegal, and the fit test checks that it collides. |
| D5 ToF cone | **Measured: use 15°.** At 27° (ROI 16×16) the cone's edge reaches the side walls of a 170 mm corridor from about 350 mm, so far walls read 300–400 mm. With no noise at all, 831 of 1,705 bank cells give a wrong digit at R = 2. At 15° (ROI 4×4) none do, and the worst reading is 54 mm (about 6σ) from a rounding boundary. Still to check against one real sensor. |
| D6 package | The mecanum robot lives in `sim/` next to the point robot, behind the same `MujocoGridEnv`, instead of in a new `mecanum_sim/` package. |

Fit checks (`tests/test_mecanum.py`):

1. Spawning at every cell of every bank maze makes no contact.
2. and 3. From every cell of every 5×5 maze, each of the 8 one-cell moves collides exactly when `can_step` says it is blocked. Otherwise it lands within 2 mm of the target centre, with noise off.
4. Doesn't apply (D4).
5. Noise-free ToF gives exactly `sensing.noiseless_signatures` in every cell of every bank maze, and no reading is under 50 mm.

## Team decisions (23.11.2026)

1. **No corner posts.**
   - The MJCF has only the walls in `wall_map`.
   - Each wall is one cell plus one wall thickness long, so walls that meet at a corner close it, and a corner with no wall stays empty.
   - A diagonal therefore works exactly as `motion_model.can_step` (D2) says, for both robots.
2. **The sensor never returns "nothing".**
   - A ray with no target in range reads the maximum range, 1.3 m. "No wall within R" is decided when the reading is discretized.
   - A miss is a ray that doesn't return properly: with probability `p_bad_return` it gets extra noise N(0, `bad_return_sigma`), and it discretizes to a wrong wall distance (closer or farther).
   - `calibrate.p_miss` reports the per-ray rate of wrong digits and the confusion matrix `confusion[true][observed]`.

## m1.4 results

100 headless episodes, `projecta`, default experiment config, noise off unless stated:

| Robot | Maze | Workers | Time | Outcomes |
|---|---|---|---|---|
| mecanum | small/m5_a (5×5) | 1 | 31 s | 100 success |
| mecanum | medium/m10_a (10×10) | 4 | 13 s | 100 success |
| mecanum, all noise on, no corrections | small/m5_a | 1 | 37 s | 73 success, 25 collision, 2 wrong declaration |
| mecanum, motion noise only (radius, strafe, slip), all corrections (default) | small/m5_a | 1 | about 20 s | 100 success |
| mecanum, all noise on, all corrections (default) | small/m5_a | 1 | about 20 s | 85 success, 15 collision |
| mecanum, all noise on, all corrections (default) | medium/m10_a | 4 | about 20 s | 77 success, 22 collision, 1 wrong declaration |

With all three corrections on, every remaining collision on 5×5 is a move that `can_step` forbids from the true cell. That means the policy chose to drive into a wall because its belief was wrong (see `sensing.py` below), not because the controller missed.
| point | small/m5_a | 1 | 2 s | 100 success |

A mecanum action takes about 30–50 ms of wall time (1 ms physics with the controller in Python), against about 10 ms for the point robot. The target was 10 minutes.

## Known issues (placeholder noise)

- **Strafe shortfall: solved by the corrections.** With the 0.90 strafe efficiency and nothing else, each sideways move ends 18 mm short against a 30 mm clearance.
  - With no corrections, that gives 16% collisions on 5×5.
  - With wall centering only, 11%.
  - With strafe compensation and front-wall alignment as well, 1 collision and 1 timeout in 100 episodes, and 0 collisions with all motion noise together.
- **`sensing.py` doesn't model noisy readings yet.** D6 lets a miss only skip a wall. Under decision 2 a reading can also come out closer, so the belief filter sees "impossible" observations. With ToF noise and bad returns, 8–12% of 5×5 episodes collide, all of them belief errors (moves that `can_step` forbids from the true cell). This is now the main source of failures. The fix belongs in `sensing.py` (t4): fit its observation model to the confusion matrix from `calibrate.p_miss`.
- **Point robot friction.** The guide's odometry integrates the commanded velocity, so with friction on the point robot stalls short without knowing it. The mecanum robot's encoders replace this.

## Changes from the setup guide

1. **`wall_map` format:** the repo uses `(rows, cols, 4)`, where 0 means a wall and 1 means open. Maps whose two sides of an edge disagree are rejected.
2. **Observations follow `sensing.py`:** the rays are N, E, S, W, each digit is 0..R or R + 1, and `observe_index()` packs them with `sensing.encode_observation`.
3. **Walls are 10 mm** (the mecanum brief), not IEEE's 12 mm.
4. **`dirichlet_summary`:** single-outcome contexts get [1, 1] instead of `nan`.
5. **Calibration contexts:** diagonals that D2 forbids are `blocked` and kept out of `free`.
6. **Calibration output** goes to `build/` by default. Write to `lidar_sim/configs/calibrated.yaml` only with measured values.

## Not done yet

- **Mecanum robot:**
  - Level B (physical rollers) and `compare_levels.py`, later by team decision.
  - Recentring after a bump (episodes end on a collision, so only scripted sequences would need it).
- **Videos:** offscreen MP4 recording (m4.3) through `on_step` and `mujoco.Renderer`.
