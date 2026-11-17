# Lidar Pygame Simulator

A grid-based simulator for developing and testing a probabilistic localization and autonomous navigation system. The robot carries five ToF (Time-of-Flight) distance sensors and uses a Bayes filter to estimate its position on the map, then plans an optimal path to a goal using A*.

---

## Running the Simulator

```bash
cd lidar_sim/
pip install -r requirements.txt
python3 lidar_pygame_sim.py
```

---

## Controls

| Key | Action |
|---|---|
| `W / A / S / D` | Move robot (up / left / down / right) |
| `Q / E` | Rotate robot counter-clockwise / clockwise (45° steps) |
| `1 – 5` | Toggle individual sensors on/off |
| `R` | Start / stop autonomous mode (localize → plan → navigate) |
| `B` | Toggle map builder mode (click cell edges to add/remove walls) |
| `P` | Toggle analysis mode (click cells to inspect their sensor data) |
| `[ / ]` | Decrease / increase auto-step delay |
| Left-click (map) | Teleport robot to that cell |
| Right-click (map) | Set navigation goal |

---

## Module Overview

```
lidar_sim/
│
├── config.py            # All constants (map size, sensor specs, noise model, thresholds)
├── discrete_noise.py    # Gaussian-derived discrete transition probabilities
├── sim_robot.py         # Robot class (sensors, movement, ray-casting)
├── map_builder.py       # Map definition, wall editing, debug data dump
├── localization.py      # Bayes filter localization algorithm
├── active_localization.py # Information-gain action selection for localization
├── motion_planning.py   # A* path planning wrapper
├── astar_planner.py     # Standalone A* planner implementation
├── rrt_planner.py       # Grid utilities: BFS, path validation, wall helpers
├── rrt_demo.py          # Console verification script for the planner
├── lidar_pygame_sim.py  # Main entry point — pygame loop, rendering, and input
└── requirements.txt     # Python dependencies
```

---

## Module Descriptions

### `config.py`
Single source of truth for every constant in the project. Nothing here is computed at runtime — change a value here and it propagates everywhere.

Key groups:
- **Display** — window size, grid size, pixel dimensions
- **Physics/Sensor** — cell size in mm, sensor range, noise level, sensor angles
- **Discrete noise model** — forward/turn tolerances, sigmas, biases, random seed
- **Wall indices** — `WALL_UP/DOWN/LEFT/RIGHT` (used throughout the codebase)
- **Algorithm thresholds** — localization confidence levels, step delays
- **Colors** — all pygame color tuples

### `sim_robot.py` — `Robot`
Represents the simulated robot. Owns its grid position, heading, and sensor state.

| Method | Description |
|---|---|
| `move_to(r, c)` | Teleport to a cell, update pixel coords |
| `move_rel(dr, dc, wall_map)` | Step one cell, blocked by walls. Returns `True` on success |
| `rotate(direction)` | +1 = clockwise, -1 = counter-clockwise (45° steps) |
| `toggle_sensor(idx)` | Enable/disable one of the 5 sensors |
| `measure(wall_map)` | Ray-cast all active sensors; returns distances in mm and hit points in pixels |

The `measure` method steps a ray along each sensor's absolute angle (robot heading + sensor offset), crossing cell boundaries until it hits a wall or reaches `MAX_RANGE_MM`. Distances are converted from pixels to mm and optionally perturbed with Gaussian noise to simulate the real VL53L4CD sensor.

### `map_builder.py`
Everything related to defining and editing the map.

| Function | Description |
|---|---|
| `generate_wall_map()` | Builds the Alon House map as a `(Rows, Cols, 4)` NumPy array |
| `toggle_wall_click(wall_map, mx, my)` | Flips the wall edge nearest to a mouse click; syncs both sides of the shared edge |
| `dump_selected_cells(data, cells)` | Prints pre-computed sensor readings for selected cells to the console (debug) |

The wall map encodes each cell's four edges as `0 = wall` / `1 = open`. Every function that reads or writes walls uses the indices from `config.py` (`WALL_UP`, `WALL_DOWN`, `WALL_LEFT`, `WALL_RIGHT`).

### `localization.py`
Implements the probabilistic localization pipeline (histogram/Bayes filter).

| Function | Description |
|---|---|
| `initialize_belief(rows, cols)` | Creates a uniform `(row, col, theta)` belief with shape `(rows, cols, 8)` |
| `precompute_all_orientations(wall_map)` | Ray-casts a virtual robot at every (cell, angle) and caches the results |
| `update_probability(prob, measured, expected)` | **Correction step** — weights every `(row, col, theta)` state by Gaussian likelihood |
| `predict_forward(prob, wall_map)` | **Prediction step** — distributes mass over undershoot/correct/overshoot forward outcomes |
| `predict_turn(prob, direction)` | **Prediction step** — distributes mass over under/correct/over turn outcomes |
| `predict_action(prob, action, wall_map)` | Dispatches `"FORWARD"`, `"TURN_LEFT"`, and `"TURN_RIGHT"` predictions |
| `collapse_position_belief(prob)` | Sums out theta for drawing the 2D heatmap |
| `get_best_estimate(prob)` | Returns `(row, col, theta, peak_probability)` of the most likely state |
| `do_localization_step(robot, wall_map, spin, moves)` | Drives the spin-then-explore recovery sequence one frame at a time |

The localization belief is fully discrete over `(row, col, theta)`, where `theta` has 8 possible headings. The sensor correction evaluates likelihood over all orientations instead of assuming the robot heading is known exactly.

**Filter sigma:** The Gaussian likelihood uses `SENSOR_LIKELIHOOD_SIGMA_MM` from `config.py`. The real sensor noise can be toggled separately with `ENABLE_SENSOR_NOISE`; the likelihood sigma is the main localization uncertainty model.

**Motion model:** Motion remains discrete. Forward and turn commands use Gaussian-derived three-bin probabilities: undershoot, correct, and overshoot. `FORWARD_TOLERANCE_MM` and `TURN_TOLERANCE_DEG` in `config.py` define the success windows and are the primary tuning knobs.

### `active_localization.py`
Replaces the old spin-then-random-walk recovery with an information-theoretic policy. When the robot is lost or its belief is multimodal, passive re-measurement cannot break the tie — every competing hypothesis predicts the same reading, so the belief stays stuck. Active localization deliberately picks the action whose *expected* observation differs most across the competing hypotheses, collapsing the belief as fast as possible.

**Theory.** The belief's uncertainty is measured by Shannon entropy (in nats):

```
H(Bel) = -sum_x Bel(x) * ln Bel(x)
```

A fully localized belief has H ≈ 0; k equiprobable cells give H = ln k. For each candidate action `a`, the policy:
1. Runs the motion model to get the predicted belief `Bel^-` (moves and turns use the discrete action prediction helpers).
2. For each of the top-K most probable hypotheses `x'`, looks up the noiseless sensor reading the robot would get if it were truly at `x'` (from `expected_data`), runs the Bayes correction, and measures the resulting entropy.
3. Averages those entropies weighted by `Bel^-(x')` to get `E[H | a]`.
4. Chooses `a* = argmax_a [ H(Bel) - E[H|a] - cost(a) ]`.

Costs are in nats so they trade off directly against information gain. Defaults are small, so during the lost phase information dominates.

| Function | Description |
|---|---|
| `select_best_action(belief, expected_data, wall_map, heading)` | Scores all 11 candidate actions and returns the best one plus diagnostics |
| `apply_action_to_robot(robot, action, wall_map)` | Executes one frame of an action: one 45° turn toward the target heading, or the forward step once aligned. Returns `(dr, dc)` moved. |
| `belief_entropy(belief)` | Shannon entropy of a probability matrix, in nats |
| `information_gain(belief, action, ...)` | Expected entropy reduction for a single candidate action |

The pygame loop calls `select_best_action` once per completed action to pick the next one, and `apply_action_to_robot` every frame to execute it step by step.

### `motion_planning.py`
A thin wrapper that calls into `astar_planner.py` and returns a path in a format the simulator loop can directly consume.

```python
path = plan_path(start=(r, c, theta_idx), goal=(r, c), wall_map=wall_map)
# Returns [(r0,c0), (r1,c1), ...] from start to goal inclusive,
# or [] if no path was found.
```

### `astar_planner.py`
Standalone A* planner implementation. Has no dependency on pygame or the simulator state.

**Key design rule:** the planner outputs *waypoint positions* (`result["path"]`), not movement commands. The simulator's navigation loop is responsible for turning the robot toward each waypoint and stepping forward.

The planner searches over `(r, c, theta_idx)` states. Action costs are 1.0 for a forward move, 0.25 for a 45° turn, and 0.5 for a 180° turn. The heuristic is Manhattan distance, which is admissible since every forward step costs ≥ 1.0. This guarantees the returned path is cost-optimal.

Key functions:

| Function | Description |
|---|---|
| `plan_astar(wall_map, start, goal, ...)` | Main entry point — returns a result dict with `"path"`, `"path_with_theta"`, and `"debug"` |
| `path_cells_to_world(path)` | Converts `[(r,c), ...]` to physical `[(x_m, y_m), ...]` coordinates (re-exported from `rrt_planner`) |

### `rrt_planner.py`
Grid utility library shared by the planner and other modules. Has no dependency on pygame.

Key functions:

| Function | Description |
|---|---|
| `bfs_shortest_path_cells(wall_map, start, goal)` | BFS shortest path between two cells; used for reachability checks |
| `validate_path(wall_map, start, goal, path)` | Checks that every step in a path is adjacent and wall-free |
| `can_move_forward(wall_map, state)` | Returns whether a forward step from `(r, c, theta)` is unblocked |
| `choose_localization_improving_waypoint(belief, wall_map, pos)` | Suggests a nearby cell with a distinctive wall signature to improve localization |

### `lidar_pygame_sim.py`
Entry point. Contains `main()` and nothing else — all logic lives in the modules above.

Responsibilities:
- Initialize pygame, fonts, and display
- Run the event loop (keyboard, mouse)
- Dispatch to `select_best_action`, `apply_action_to_robot`, `plan_path`, `predict_action`, `update_probability` at the right times
- Manage the auto-mode state machine (`IDLE → LOCALIZING → PLANNING → MOVING → DONE`)
- Draw the map, probability heatmap, robot, sensor rays, planned path, and sidebar

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     lidar_pygame_sim.py                     │
│                                                             │
│  Events ──► Robot.move_rel / rotate / apply_action          │
│                                                             │
│  Each frame:                                                │
│    Robot.measure(wall_map)                                  │
│         │                                                   │
│         ▼                                                   │
│    update_probability(prob, dists, expected_data)           │
│         │                                                   │
│         ▼                                                   │
│    predict_action(prob, action, wall_map)  (if commanded)   │
│         │                                                   │
│         ▼                                                   │
│    get_best_estimate(prob)                                  │
│         │                                                   │
│         ├─ peak < threshold ──► select_best_action(belief)   │
│         │                           │                       │
│         │                           ▼                       │
│         │                    apply_action_to_robot()        │
│         │                    (one frame: turn or move)      │
│         │                                                   │
│         ├─ peak ≥ threshold ──► plan_path(start, goal, map) │
│         │                           │                       │
│         │                           ▼                       │
│         │                      planned_path[]               │
│         │                           │                       │
│         └─────────────────────────► follow path step-by-step│
└─────────────────────────────────────────────────────────────┘

Supporting data structures:
  wall_map      (Rows, Cols, 4)  numpy int array — the map
  expected_data (Rows, Cols, 8, 5) numpy float array — precomputed sensor readings
  prob_matrix   (Rows, Cols, 8)  numpy float array — localization belief
```

---

## Map Format

The map is a NumPy array of shape `(Rows, Cols, 4)`.

- `wall_map[r, c, WALL_UP]` — top edge of cell `(r, c)`
- `wall_map[r, c, WALL_DOWN]` — bottom edge
- `wall_map[r, c, WALL_LEFT]` — left edge
- `wall_map[r, c, WALL_RIGHT]` — right edge

`0 = wall present`, `1 = passage open`.

Every wall is stored on both sides of the shared edge (e.g., the right wall of `(r, c)` equals the left wall of `(r, c+1)`). `map_builder.py` keeps these in sync.

## Heading Convention

Angle index `0` points East; indices increase clockwise at 45° steps.

| Index | Direction |
|---|---|
| 0 | East |
| 1 | South-East |
| 2 | South |
| 3 | South-West |
| 4 | West |
| 5 | North-West |
| 6 | North |
| 7 | North-East |

The A* planner only moves on cardinal headings (0, 2, 4, 6). Diagonal headings are used only for rotation.
