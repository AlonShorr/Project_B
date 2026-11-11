# Kidnapped Robot Project (Project B): Technical Roadmap

> **For LLM assistants:** This is the engineering companion to `ROADMAP.md`. `ROADMAP.md` says *what* is delivered and *when*: phases, owners, weeks, and acceptance criteria with IDs like `t3.2`. This file says *how*: the exact models, interfaces, math, code reuse from Project A, tests and pitfalls.
> - Schedule and acceptance criteria: `gantt-project-b/public/plan.json` wins, then `ROADMAP.md`.
> - Technical design: this file wins until a decision in §2 is changed. Record every changed decision in §2 with the date.
> - Items marked **[NEW]** are not in `plan.json` yet. If the team accepts one, add it to `plan.json` with a new task ID (`t20`, `t21`, …). Never reuse an existing ID.

---

## 0. Where we start from (Project A inventory)

### 0.1 Where the Project A code lives

The latest Project A code is the Git repo **`AlonShorr/Project_A`** (GitHub, `git@github.com:AlonShorr/Project_A.git`). Segev's clone is in WSL at `\\wsl.localhost\Ubuntu\home\logan\Project_A`. The last commit checked is `5f5f98e` (2026-07-02, ".gitignore update").

| Path in `Project_A/` | What it is | Use in Project B |
|---|---|---|
| `development/SA_Simulator/lidar_sim/` | **Project A's final simulator ("v2")**: the Pygame app, Bayes filter, active localization, A\* and map generator. Python 3.12, `numpy`, `matplotlib`, `pygame` | **Main code base to port** (§0.2) |
| `development/SA_Simulator/local_sim.py`, `robot.py`, `maps.py` | Simulator v1 (Matplotlib prototype, 2-D belief) | History only |
| `development/SA_Simulator/README_RRT.md`, `lidar_sim/rrt_planner.py`, `rrt_demo.py` | Alon's RRT planner (replaced by A\*). `rrt_planner.py` still holds shared grid helpers (`_action_cost`, `apply_internal_action`, `bfs_shortest_path_cells`, `validate_path`, …) | Reuse the helpers if they help, not RRT itself |
| `development/localization/` | ESP32 (ESP-IDF) C++ histogram localization with `WifiLogger`. Interfaces in `development/components/micromouse_interfaces/` (`localization_port.h`) | Reference only (the physical robot is out of scope) |
| `legacy/micromouse-py-sim-main/` | The predecessor team's simulator (Abudi & Huly), the same code as `Semester_7/ProjectA/micromouse-py-sim-main` | Maze files and the `.maze` loader only (§0.3) |
| `legacy/micromouse-robot-main/` | Predecessor ESP32 firmware | Robot geometry for MuJoCo (`m1`) |

**Action item `t0a`:** the code has been found. Add Project_A to the Project B repo as a read-only reference (`git subtree` or a pinned submodule under `third_party/Project_A`). Port code into `kr/` by copying and refactoring; do not import `lidar_sim` directly. It depends on module-level globals (`config.MAP_ROWS`, …) and on Pygame being importable.

### 0.2 What `lidar_sim` actually does (read from the code)

**Data structures** (numpy):
- `wall_map`: shape `(rows, cols, 4)`, int. Channels `WALL_UP=0, WALL_DOWN=1, WALL_LEFT=2, WALL_RIGHT=3`. **`0 = wall`, `1 = open`.** Both sides of a shared edge are stored and kept in sync (`map_generator._set_edge`, `map_builder.toggle_wall_click`).
- `prob_matrix`: belief of shape `(rows, cols, 8)`. Heading index 0 = East, increasing clockwise in 45° steps (2 = S, 4 = W, 6 = N).
- `expected_data`: shape `(rows, cols, 8, 5)`, the noiseless ray-cast reading (mm) for every pose and sensor. Built by `precompute_all_orientations`.

**Map size is hard-coded.** `config.py` derives `MAP_COLS = 800 // 40 = 20` and `MAP_ROWS = 600 // 40 = 15` from the window size. `Robot.move_rel`, `map_generator` (`_base_map`, `validate_wall_map`, `score_map`) and `initialize_belief` callers all use these globals, so **5×5, 10×10 and 16×16 mazes cannot run without refactoring.**

**Sensors** (`sim_robot.Robot.measure`):
- 5 rays at `SENSOR_ANGLES = [-90, -40.33, 0, 37.22, 90]`° relative to heading.
- Rays are marched in 2-pixel steps (40 px per 180 mm cell) and stop at the first closed edge or at 1300 mm.
- Optional Gaussian noise σ = `SENSOR_MEASUREMENT_SIGMA_MM` = 1 mm, drawn with Python's **global `random.gauss`, not the seeded numpy `rng`.**

**Sensor model** (`localization.update_probability`):
- `∏_i exp(−(z_i − max(ẑ_i, 1mm))² / 2σ²)` with σ = `SENSOR_LIKELIHOOD_SIGMA_MM` = 20 mm. Sensors with `None` (disabled) are skipped.
- Each state's likelihood is floored at `1e-9`, so no state ever reaches zero. If the total mass is 0, the belief resets to uniform.

**Motion model** (`discrete_noise.py`, `localization.predict_forward` / `predict_turn`, `Robot.apply_action`):
- Actions are `FORWARD`, `TURN_LEFT` and `TURN_RIGHT` (±45°). `FORWARD` only moves on cardinal headings 0/2/4/6. On a diagonal heading it does nothing.
- Each action has three outcomes: under, correct and over (0, 1 or 2 cells, or 0, 1 or 2 turn steps). The probabilities come from a Gaussian over the target with a ±half-bin window, plus explicit extra mass:
  - forward: target 180 mm, σ = 15 mm, half-bin 90 mm, extra stay 0.02, extra overshoot 0.01, so **≈ (0.02, 0.97, 0.01)**
  - turn: target 45°, σ = 4°, half-bin 22.5°, extra 0.02 each side, so **≈ (0.02, 0.96, 0.02)**
- **A wall stops the robot in front of it. There is no collision.** `_advance_one_cell` returns the same cell, and `move_forward_noisy` breaks out of its loop.
- `ENABLE_MOTION_NOISE`, `ENABLE_SENSOR_NOISE` and `RANDOM_SEED = 42` are global flags.

**Active localization** (`active_localization.select_best_action`):
- **11 candidate actions:** rotate to any of the other 7 headings, or "move" in one of 4 cardinal directions (turn to that heading, then one forward step).
- Score: `gain − cost`, where `gain = H(b) − E[H(b⁺)]` in nats. Costs: `move_cost = 0.10` per move, `turn_cost = 0.02` per 45° step.
- `E[H]` is approximated over the **top 16 hypotheses** (those with probability ≥ `1e-4`). Each hypothesis's noiseless reading is used as the observation and weighted by its probability. Each hypothesis runs a full `update_probability`.
- The chosen action runs one turn per frame (`apply_action_to_robot`) and is re-selected once the target heading is reached.

**Localization gate and state machine** (`lidar_pygame_sim.main`):
- States `IDLE → LOCALIZING → PLANNING → MOVING → DONE`.
- Localized when `peak ≥ LOCALIZATION_THRESHOLD = 0.25` **and** `runner_up ≤ 0.5·peak`. Here `runner_up` is the best position (summed over headings) at Chebyshev distance ≥ 2 from the peak (`dominant_position_modes`).
- While MOVING, the robot re-localizes when the peak state probability falls below `IS_LOST_THRESHOLD = 0.15`, when the next waypoint is not adjacent, or when a forward move is blocked. **`reset_auto` does not reset the belief**, so localization resumes from the current belief.
- **`DONE` is triggered by the true robot position** (`(robot.r, robot.c) == goal`), not by a belief decision. There is no "declare" action.
- The robot starts at the map's fixed `start` cell, while the belief starts uniform.

**Planner** (`astar_planner.plan_astar`, wrapped by `motion_planning.plan_path`): A\* over `(r, c, θ)` from the MAP estimate. Forward costs 1.0, a 45° turn 0.25, a 180° turn 0.5, with a Manhattan heuristic. The output is cell waypoints, and the main loop turns and steps toward each one.

**Maps** (`map_generator.get_maps`): 10 maps of 15×20, selected with F1–F10.
- 5 hand-designed maps: Training Maze, Crossroads, Room Grid, Serpentine, Chambers.
- 5 "Photo Grid" maps: a DFS recursive backtracker (`_dense_photo_maze(seed, start, goal, loop_edges)`) with hand-picked loop edges.
- `validate_wall_map` checks the borders, edge symmetry and full connectivity.
- `score_map`:
  - `ambiguity_score`: the number of cells whose unbounded 4-direction wall-distance signature is shared with another cell
  - `shortest_path_steps`: BFS distance from start to goal
  - Maps are ranked easy/medium/hard by the sum of the two.

**Performance:** `update_probability`, `predict_forward` and `predict_turn` are **pure-Python triple loops** over `rows × cols × 8` (2,400 states on 15×20). One active-localization decision calls `update_probability` up to 11 × 16 = 176 times, so it takes seconds per step. This is fine for the GUI and **too slow for thousands of batch runs.**

**Tests:**
- `test_discrete_noise.py`: the noise probabilities sum to 1, plus sampling checks.
- `python3 active_localization.py`: a self-test demo (entropy sanity checks, positive gain from a uniform belief, a two-mode belief).

Evaluation in Project A was qualitative only. There was no batch runner, no CSV output and no failure-rate measurement.

### 0.3 The predecessor simulator (`legacy/micromouse-py-sim-main/sim`)

This is the earlier team's exploration simulator (package `sim`, Python ≥ 3.12). The robot is **told its true pose** every step (`Simulator.step` sends `RobotState`), motion is deterministic, and a wall hit sets `SimulationStatus.ERROR`. It is useful to us only for:
- `sim.maze.Maze.from_file` loaders (`.maze`/`.num`/`.maz`/`.csv`) and `Walls` bit flags (N=1, E=2, S=4, W=8)
- the 16×16 competition mazes `japan_2017.maze` and `semifinal_2009.maze` (goal = center 2×2), and the 5×5 mazes `simple`, `longer`, `floating*`

Write one converter, `maze_to_wall_map(Maze) -> np.ndarray(rows, cols, 4)`. Use `wall_map` everywhere else.

### 0.4 Gap analysis: Project A `lidar_sim` against the Project B problem definition

| Project B requirement | Project A `lidar_sim` today | Required change | Task |
|---|---|---|---|
| Any maze size (5×5, 10×10, 16×16) | 15×20 fixed by display constants | Pass `rows, cols` explicitly. Remove the globals from all logic | `t5` |
| State without heading (sensing needs no rotation) | Belief `(r, c, 8)`. Rotations are actions | Belief `(r, c)` → **8× smaller**. Drop turn actions, turn noise and heading A\* | `t3`, `t4` |
| 8 move directions, no rotation | FORWARD only on cardinal headings, ±45° turns | Absolute moves N, NE, E, SE, S, SW, W, NW | `t3` |
| Motion noise 0/1/2 steps **and drift to adjacent directions** | 0/1/2 steps along the heading; no lateral drift | Keep the Gaussian three-bin parametrization for k. Add `p_drift` | `t3` |
| **Collision ends the run** | Wall stops the robot in place | `COLLISION` terminal outcome; `C[a, s]` probability | `t3.3` |
| Range-R obstacle sensing with misses | 5 continuous ToF rays to 1300 mm, Gaussian likelihood | Discrete rays with range R and `p_miss` (§3.3). Keep the Gaussian ToF model as an **optional** high-fidelity mode for MuJoCo | `t4` |
| **Declare "arrived"** | `DONE` when the true pose reaches the goal | `DECLARE` action decided from the belief. Success iff the true cell ∈ G | `t3.0`, `t7` |
| Seeded, reproducible batch runs | GUI loop. Global `random` for sensor noise and `do_localization_step` | Headless environment. One `np.random.Generator` per run | `t5.1` |
| ≥ 1,000 runs in minutes | Python loops; seconds per decision | Vectorized numpy: `b @ T[a]`, `Z[:, o] * b̄`, batched lookahead | `t5.2` |
| Mazes per size with ambiguity scores | 10 maps of 15×20 with `score_map` | Generalize the generator and `score_map` to any size and to R-limited signatures | `t6` |
| Pareto evaluation | None | Runner + CSV + analysis | `t5`, `t8` |

**Main simplification against Project A:** sensing needs no rotation and actions are absolute directions, so heading disappears from the state. The belief shrinks from `rows×cols×8` to `rows×cols` (25 / 100 / 256 states), and the turn actions, turn noise and heading-based A\* all go away. This is what makes exact and point-based POMDP solving feasible.

**What carries over almost unchanged:**
- the Bayes filter structure (predict, then correct, then normalize with the uniform fallback)
- the entropy and expected-information-gain policy (`belief_entropy`, the `gain − cost` scoring)
- the ambiguity-aware gate (`dominant_position_modes` with ratio 0.5 and separation 2)
- the three-bin Gaussian noise helpers (`discrete_noise.gaussian_three_bin_probs`, `mix_extra_outcomes`), which also parametrize the MuJoCo calibration targets
- the DFS-with-loops maze generator, `validate_wall_map` and `score_map`

---

## 1. Target code architecture (Project B repo)

Create a new repo (for example `kidnapped-robot/`) next to `gantt-project-b/`. Keep `AlonShorr/Project_A` as a read-only reference under `third_party/Project_A` (§0.1). **The native map format is Project A's `wall_map` array** `(rows, cols, 4)`, with `0 = wall` and `1 = open` (§0.2). The predecessor `sim.maze` is used only to import `.maze` files.

```
kidnapped-robot/
  configs/
    default.yaml            # every parameter (t5.0); experiments override by file or CLI
    experiments/*.yaml
  mazes/                    # wall_map .npz (+ .maze text for humans), loaded by name (t6.2)
    small/  medium/  large/
  kr/                       # python package
    model.py                # Model: the single source of truth for T, Z, C, goal set (§3)
    walls.py                # wall_map helpers ported from lidar_sim (_set_edge, validate, BFS), maze_to_wall_map()
    noise.py                # ported discrete_noise.py (Gaussian three-bin probabilities)
    maze_tools.py           # generator + score_map ported from lidar_sim/map_generator.py, any size (t6)
    env_grid.py             # GridEnv: samples from Model (t3, t4, t5)
    env_mujoco.py           # MujocoEnv: same Env protocol (m1–m3)
    mjcf.py                 # wall_map -> MJCF XML (m1.0)
    belief.py               # exact discrete Bayes filter over Model (t4.1)
    policies/
      base.py               # Policy protocol
      projecta.py           # ported Project A heuristic, vectorized (t7)
      qmdp.py               # [NEW] cheap POMDP baseline
      alpha.py              # alpha-vector policy from a solver (t13)
      pomcp.py              # fallback online solver (t15)
    pomdp_io.py             # write Cassandra .pomdp, read SARSOP .policy (t10.1, t13.0)
    runner.py               # headless batch runner -> CSV (t5)
    analysis/
      pareto.py             # frontier, convex hull, CIs (t8, t14, t17)
      plots.py
  scripts/
    solve_sarsop.sh         # one-command reproduction (t12.2)
    sweep_projecta.py  sweep_lambda.py  calibrate_mujoco.py
  tests/
  viz/                      # optional: Pygame viewer adapted from lidar_pygame_sim.py (belief heatmap, replay a CSV run)
```

### 1.1 Core interfaces

These interfaces are the contract that lets one policy run on both simulators (`m1.3`, `t13.1`, `m4.0`).

```python
# kr/model.py
Cell = int                       # index into free cells, 0..N-1
class Act(IntEnum):              # 8 moves + declare
    N=0; NE=1; E=2; SE=3; S=4; SW=5; W=6; NW=7; DECLARE=8
Obs = int                        # index into the finite observation space O

@dataclass(frozen=True)
class Model:
    wall_map: np.ndarray         # (rows, cols, 4), 0 = wall, 1 = open  (Project A convention)
    cells: np.ndarray            # (N, 2) row/col of each free cell
    goal: np.ndarray             # bool (N,)
    T: np.ndarray                # (A_move, N, N)  P(s' | s, a, no collision)
    C: np.ndarray                # (A_move, N)     P(collision | s, a)
    Z: np.ndarray                # (N, |O|)        P(o | s)   (observation after arrival)
    obs_decode: Callable[[Obs], tuple]  # for debugging and plots
    params: ModelParams          # p_steps, p_drift, R, p_miss, ...
    # invariant: T[a, s].sum() + C[a, s] == 1 for all a, s   (t3.1)

# kr/env_grid.py, kr/env_mujoco.py
class Outcome(Enum): RUNNING, SUCCESS, WRONG_DECLARATION, COLLISION, TIMEOUT
class Env(Protocol):
    def reset(self, seed: int, start: Cell | None = None) -> Obs: ...   # start ~ Uniform(free) if None
    def step(self, a: Act) -> tuple[Obs | None, Outcome, dict]: ...    # info: true cell, steps, ...

# kr/policies/base.py
class Policy(Protocol):
    def reset(self, model: Model, b0: np.ndarray) -> None: ...
    def act(self, obs: Obs) -> Act: ...        # the policy keeps its own belief
```

**Rule:** `Model` is built once from `(maze, params)`. `GridEnv` *samples* from it, `belief.py` *filters* with it, and `pomdp_io.py` *exports* it. Never write the dynamics twice. This guarantees that the simulator, the filter and the solver agree, and it removes a whole class of bugs.

### 1.2 Configuration (`t5.0`)

YAML file, loaded into frozen dataclasses. Every run writes its fully resolved config hash into the CSV.

```yaml
maze: {name: small/m5_a, goal: [[2,2]]}
motion: {step_sigma_mm: 15, step_bias_mm: 0, extra_stay: 0.02, extra_over: 0.01,   # Project A defaults -> p_steps
         p_steps: null,                     # or give (p0, p1, p2) directly; overrides the Gaussian form
         p_drift: 0.1, diag_rule: both_L_open, wall_mode: collide}
sensing: {R: 2, rays: 4, p_miss: 0.05, miss_mode: skip_to_next}
episode: {max_steps: 200, declare_cost: 0}
projecta: {tau_loc: 0.25, rho_amb: 0.5, tau_lost: 0.15, tau_declare: 0.95, eps_col: 0.01, move_cost: 0.1}
pomdp: {lambda: 50, gamma: 0.999, solver: sarsop, precision: 0.01, timeout_s: 3600}
mujoco: {enable_slip: true, enable_ctrl_noise: true, enable_sensor_noise: true, ...}
run: {n_runs: 1000, seed0: 0, out: results/xyz.csv}
```

---

## 2. Design decisions (record changes here)

| ID | Decision | Default (recommended) | Why / alternatives | Status |
|---|---|---|---|---|
| D1 | World geometry | **Thin walls on cell edges**, stored as Project A's `wall_map` `(rows, cols, 4)`, not occupied cells | Matches Project A, the predecessor mazes, MuJoCo and the robot. The supervisor's text says "grid with obstacles". Confirm in `t1` | Proposed |
| D2 | Diagonal move legality | Diagonal NE from (r,c) is legal iff **both** L-paths N→E and E→N are wall-free. Otherwise it is a collision | Conservative and symmetric. Alternative: "at least one L-path open" (corner cutting) | Proposed |
| D3 | Multi-step semantics | k ∈ {0,1,2} cells along the realized direction. Each traversed edge is checked. A wall at step 1 or 2 means `COLLISION` (no partial stop) | Matches "collision stops the run". **This changes Project A**, where the robot silently stops in front of the wall (`_advance_one_cell`). Keep Project A's behavior as a config switch (`wall_mode`: `collide` or `stop`) for ablations | Proposed |
| D4 | Drift | With `p_drift`, the realized direction is ±45° from intended (split equally). Applies to diagonals too | Matches "adjacent directions" | Proposed |
| D5 | Observation | **4 cardinal rays** (optionally 8). Each ray reports the distance in cells to the first wall, `0..R`, or `NONE` if beyond R. Independent across rays | Finite \|O\| = (R+2)^rays. With R=2 and 4 rays: 256. With 8 rays: 65,536, which is too many for the solver | Proposed |
| D6 | Miss model | Each wall within range is missed independently with `p_miss`. The ray then reports the next unmissed wall within range, or `NONE` | Simple and exact to enumerate. Alternative: a miss only on the first wall | Proposed |
| D7 | When observations happen | One observation after every move (and one at reset). `DECLARE` gives no observation | Standard POMDP timing: `Z(o | s')` | Proposed |
| D8 | Objective | Lagrangian: minimize `E[T] + λ·P(fail)`. Sweep λ | Optimal for the constrained problem only on the **convex hull** (`t14.2`). Mixing two policies attains hull segments. For a true constrained solver see the CPOMDP references in `t2` and the `λ sweep` risk in §6 | Proposed |
| D9 | MuJoCo robot | **Holonomic planar body** (slide x, slide y; yaw locked) driven by velocity actuators | "No rotation" plus 8 directions maps naturally to holonomic motion. A differential drive would need turns that the model does not count | Needs supervisor OK (`t1.1`) |
| D10 | Start distribution | Uniform over free non-goal cells | Kidnapped robot. b₀ is fixed, so SARSOP can use reachable-belief sampling | Proposed |

Defaults to fix in `t1.3`: d ∈ {1%, 5%, 10%} (evaluation points), R = 2, `p_drift` = 0.1, `p_miss` = 0.05. For `p_steps`, Project A's defaults give ≈ (0.02, 0.97, 0.01), which is nearly deterministic. Use (0.1, 0.8, 0.1) as the "hard" setting and Project A's values as the "easy" setting. All of these are placeholders until `m3` calibrates them.

---

## 3. Formal model (shared by every component)

### 3.1 States, actions, terminal outcomes

- `S = {free cells} ∪ {⊥}`, where ⊥ is an absorbing terminal state. In code, terminal outcomes are reported by the environment and are not belief states.
- `A = {N, NE, E, SE, S, SW, W, NW, DECLARE}` (9 actions).
- Terminal outcomes: `SUCCESS` (`DECLARE` at s ∈ G), `WRONG_DECLARATION` (`DECLARE` at s ∉ G), `COLLISION` (a move hits a wall), and `TIMEOUT` (`max_steps` reached, counted as a failure, rare by design).

### 3.2 Transition model `T`, `C` (`t3`)

For a move action with intended direction `d`:

```
P(dir = d) = 1 − p_drift ;  P(dir = d ± 45°) = p_drift / 2
P(k)       = p_steps[k],  k ∈ {0, 1, 2}     # default: noise.gaussian_three_bin_probs(180, 90, σ, bias) mixed with extras (Project A)
for each (dir, k): walk k cells in dir from s; if any edge crossed is blocked (D2 for diagonals) → collision
T[a, s, s'] = Σ_{dir,k} P(dir)·P(k)·[walk ends at s' without collision]
C[a, s]     = Σ_{dir,k} P(dir)·P(k)·[walk collides]
```

Port `discrete_noise.gaussian_three_bin_probs` and `mix_extra_outcomes` unchanged into `kr/noise.py`, so the Project A parametrization (σ, bias, extra stay/overshoot) stays available and matches what MuJoCo calibration (`m3`) estimates. Build `T` as dense `(8, N, N)` float64. N ≤ 256 gives about 4 MB, which is fine. Use `scipy.sparse` only if it becomes necessary.

**Tests (`t3.1`, `t3.2`):** `T.sum(-1) + C == 1`. Empirical frequencies from `GridEnv` over 10⁴ samples per (s, a) are within ±1% of `T`/`C` (use a χ² test as well as the ±1% check). In the open interior with `p_drift=0` and `p_steps=(0,1,0)` the model must be deterministic.

### 3.3 Observation model `Z` (`t4`)

For each free cell s and ray direction d, precompute the list of wall distances within range, `w_d(s) = [δ₁ < δ₂ < …] ⊆ {0..R}`. Under D6, the ray reports `δ_j` with probability `p_miss^(j−1)·(1−p_miss)` and `NONE` with probability `p_miss^{|w_d|}`. Then `Z[s, o] = ∏_d P(o_d | s)`, with `o` encoded in mixed radix base `(R+2)`.

Note: with thin walls along a straight cardinal ray, normally only the first wall is within range unless R ≥ 2 and the first wall is missed. D6 handles this exactly.

**Tests:** `Z.sum(1) == 1`. With `p_miss=0`, `Z` is one-hot. `t4.2`: two cells with different noiseless signatures are separated by a single observation, so the posterior is zero on the wrong one.

**Ambiguity score (`t6.1`):** Project A's `map_generator.score_map` counts cells whose *unbounded* 4-direction wall-distance signature is shared. Keep it for continuity with Project A, and add the R-limited version that matches what the robot can actually sense: group free cells by noiseless signature `σ(s) = argmax_o Z[s, o]` with `p_miss=0`. Report `amb = 1 − (#distinct signatures / N)` and the size of the largest group. Also report the expected posterior entropy after one observation from b₀, `H₁ = Σ_o P(o)·H(b₀|o)`. Use these to choose mazes that span easy to hard.

### 3.4 Belief update (exact, `belief.py`)

```
predict:  b̄(s') = Σ_s b(s) T[a,s,s']  /  (1 − Σ_s b(s) C[a,s])     # condition on "not collided"
correct:  b'(s') ∝ Z[s', o] · b̄(s')
```

Condition on survival because the run ends on a collision. A policy that is still acting knows that no collision happened. Use log-space or renormalize every step. If `Σ Z·b̄ == 0` (impossible observation, possible only from model mismatch in MuJoCo), apply a **uniform-mixing fallback** `b ← (1−ε)b + ε·U` (Burgard et al. 1996), log the event, and count it in the results.

**Tests (`t4.1`):** normalization holds after each step. A filter run on `GridEnv` must assign the true cell non-zero probability at every step (property test over random seeds).

### 3.5 Reward for the POMDP (`t10`)

```
R(s, move)      = −1                    # each step costs 1 → E[T]
R(s, move) also −λ · C[a, s]            # expected collision penalty
R(s, DECLARE)   = 0 if s ∈ G else −λ
γ               = 0.999 (solver requirement; report sensitivity to γ ∈ {0.99, 0.999})
```

Then `V^π(b₀) = −(E[T] + λ·P(fail))` up to discounting. Collisions and declarations move to ⊥, which has reward 0 and is absorbing. `DECLARE` everywhere → ⊥.

---

## 4. Phase-by-phase technical plan

### Phase p0: Kickoff (wk 1–3)

- **`t1` scope email:** attach §2 (D1–D10) and ask the supervisor to confirm D1, D2, D5, D8 and D9 explicitly. The Project A switching criterion "?" becomes **`P_max ≥ τ_loc ∧ ratio < ρ_amb`** (Project A's real rule, §5.1), which `t1.2` should confirm.
- **`t2` learning, focused list:**
  - pomdp.org tutorial
  - Kaelbling, Littman & Cassandra 1998, §1–4
  - Tiger by hand
  - Then: Pineau, Gordon & Thrun 2003 (PBVI); Spaan & Vlassis 2005 (Perseus); Smith & Simmons 2005 (HSVI2); Kurniawati, Hsu & Lee 2008 (SARSOP); Silver & Veness 2010 (POMCP); Littman, Cassandra & Kaelbling 1995 (QMDP); Hauskrecht 2000 (value-function approximations, FIB)
  - For the constraint: Isom, Meyn & Braatz 2008 and Kim et al. 2011 (constrained POMDPs, CPBVI); Poupart et al. 2015 (CALP); Lee et al. 2018 (CC-POMCP)
- **[NEW] `t0b` repo bootstrap (S, wk 1):** new repo with `pyproject.toml` (Python 3.12, numpy, scipy, pyyaml, pandas, matplotlib, pytest, optional mujoco and pygame), pre-commit, and CI running pytest. Add `AlonShorr/Project_A` as `third_party/Project_A` (pinned commit) and port into `kr/`: `discrete_noise.py` → `noise.py`; the `wall_map` helpers and `validate_wall_map` → `walls.py`; `map_generator.py` → `maze_tools.py` with `rows, cols` as parameters. Port the predecessor `.maze` loader for the 16×16 mazes.

### Phase p1: Simulator and baseline (wk 1–5)

**`t3` motion model (S).** Implement `Model.build_T()` per §3.2, plus `GridEnv.step()` sampling `(dir, k)` with the env RNG.
- Start from Project A's `_advance_one_cell` and `Robot.move_rel` edge checks (`wall_map[r, c, WALL_*] == 0` means blocked). Extend them to 8 directions with D2, and change "stop at wall" into `COLLISION` (D3).
- Pitfall: in Project A, the predictor (`predict_forward`) and the true robot (`Robot.move_forward_noisy`) are separate code paths that must be kept consistent by hand. With `Model` as the single source of truth, this class of mismatch goes away.
- Pitfall: Project A draws sensor noise with the global `random` module. All randomness in `GridEnv` must come from its own `np.random.Generator`.

**`t4` sensing model (A).** Implement `w_d(s)` ray tables and `Z` per §3.3, and write the formal O(o|s) in the formalization doc (`t4.0`).
- `map_generator._wall_distance` already computes the unbounded per-direction cell distance. Cap it at R and add the miss model.
- Keep Project A's continuous ToF model (`Robot.measure` + `update_probability`, σ = 20 mm) as an optional `sensing.mode: tof5` for the MuJoCo comparison. Do not use it in the POMDP, because its observation space is continuous.

**`t5` config and headless runner (S).**
- `runner.py`: `for seed in range(seed0, seed0+n): env.reset(seed); policy.reset(model, b0); loop`.
- One CSV row per run: `run_id, seed, maze, policy, policy_params_json, T, outcome, steps_to_loc, final_P_goal, n_fallbacks, wall_time_ms, config_hash`.
- Use one `np.random.Generator(PCG64(seed))` for the environment and a **separate** generator for the policy (POMCP), so changing a policy does not change the start cells (common random numbers for paired comparisons).
- Parallelize with `multiprocessing.Pool` over seeds. Build `Model` once per worker.
- Target `t5.2`: 1,000 runs on 10×10 in under 10 min. Exact belief updates are O(N²) per step with N=100, which is trivial *once vectorized*. Project A's filter is pure-Python triple loops, and one active-localization decision runs up to 176 full updates (§0.2), so it **must** be rewritten with numpy (`b @ T[a]`, `Z[:, o] * b̄`, all observations at once).
- Remove every `print` from the hot path and use `logging`.

**`t6` maze suite (S).**
- Generator: port Project A's `_dense_photo_maze(seed, start, goal, loop_edges)` (DFS recursive backtracker on a closed `wall_map`) with `rows, cols` as parameters. Replace the hand-picked `loop_edges` with **random loop injection** (open `k%` extra walls). Loops create ambiguity and multiple routes.
- Validate with the ported `validate_wall_map` (borders, edge symmetry, full connectivity).
- Project A's 10 maps are all 15×20, so none fits the suite directly. Crop or redraw `Serpentine` and `Chambers` as hard 10×10 examples.
- Goal: center cell for odd sizes; for 16×16, the standard 2×2 center (as in `presets.json`).
- Choose 3 or more per size spanning ambiguity (§3.3).
- 16×16: also include `japan_2017` and `semifinal_2009`. 5×5: adapt `simple`/`longer`.
- Save as `wall_map` `.npz`, plus an ASCII `.maze` rendering for humans, plus a `mazes/index.yaml` with goal, size and both ambiguity scores.

**`t7` port the Project A heuristic (A).** See §5.1 for the exact algorithm. It must expose `τ_loc, ρ_amb, τ_declare, ε_col, move_cost` (`t7.0`).
- Source files: `active_localization.py` (`belief_entropy`, `select_best_action`), `localization.py` (`dominant_position_modes`), `lidar_pygame_sim.py` (the auto-mode loop), `astar_planner.py`.
- Keep Project A's `active_localization._demo()` checks as unit tests: entropy of k equal cells = ln k, positive gain from a uniform belief, and a two-mode belief gets resolved.

**`t8` Project A Pareto curve (B).**
- Grid over `(τ_declare ∈ {0.8, 0.9, 0.95, 0.98, 0.99, 0.995}) × (ε_col ∈ {0, 0.005, 0.02, 0.05})`, plus `τ_loc` and `ρ_amb` around the Project A defaults. That gives ≥ 20 configurations (`t8.0`).
- 500 runs per configuration per maze.
- Plot mean T (95% CI, bootstrap) against failure rate (95% **Wilson** interval). Highlight the non-dominated set.
- Also report the failure breakdown (collision against wrong declaration), because it tells you which knob to turn.

### Phase pm: MuJoCo environment (wk 3–9)

**`m1` maze and robot (S).**
- `mjcf.py`: for every closed edge in `wall_map`, emit a `box` geom. IEEE dimensions: cell 180 mm (same as Project A's `CELL_SIZE_MM`), wall 12 mm thick and 50 mm high, posts 12×12 mm. Emit only `WALL_UP` and `WALL_LEFT` of each cell plus the outer bottom and right border, so no wall is emitted twice.
- Robot: a cylinder or box body of about 100 mm footprint (from the predecessor robot), with `slide` joints on x and y and no free yaw (D9). Velocity actuators.
- Sensors: MuJoCo `rangefinder` sensors on sites pointing in the 4 (or 8) ray directions, cut off at `(R+0.5)·180 mm`.
- Low-level controller for a discrete action: target = current cell center + k·d·180 mm, with a P or PD velocity loop and a stop condition (speed < ε or timeout). After the stop, **discretize**: cell = round(position / 180 mm). Collision = any contact between the robot geom and a wall geom (check `data.contact`).
- Observation discretization: `o_d = floor((range − wall_offset)/180 mm)`, clipped to `0..R`, or `NONE`.
- Headless: `mujoco` Python bindings, `mj_step` without a viewer. The viewer is only for `m4.3` videos.

**`m2` uncertainty sources (S).** Each source has an on/off flag and a parameter:
- friction ~ U[μ_lo, μ_hi] per run
- multiplicative velocity-command noise
- stop-time jitter
- range noise σ(r) = a + b·r
- a dropout probability that reports `NONE`

Use the VL53L4CD datasheet (1–1300 mm, 18° FoV) and predecessor measurements as sources (`m2.4`). The goal is that `p_steps`, `p_drift` and `p_miss` **emerge** from physics instead of being injected directly.

**`m3` calibration (S).** For each action type (cardinal, diagonal) run ≥ 1,000 commands from random interior cells. Tabulate the outcome `(dir, k)` or collision and fit categorical probabilities with Dirichlet or Wilson CIs (`m3.1`). Do the same for `p_miss` (`m3.2`). Write the results to `configs/calibrated.yaml`, then **rebuild `Model`** and re-export the POMDP (`m3.3`).
- Expect the noise to depend on the state (near walls compared with open space). Record the residual mismatch. It explains the sim-to-sim gap in `m4.2`.

### Phase p2: POMDP formulation (wk 4–6)

**`t10` (A).** The written chapter states §3 formally.
- `pomdp_io.write_cassandra(model, λ, γ, path)` emits the Cassandra `.pomdp` format:
  - `discount`, `values: reward`, `states: N+1`, `actions: 9`, `observations: |O|+1` (with an extra `o_term`)
  - `start:` b₀
  - sparse `T:` and `O:` lines, plus `R:` lines
- ⊥ self-loops with observation `o_term`.
- Size check, 5×5 with R=2: 26 states × 257 observations × 9 actions. Small.
- Validate by loading in SARSOP (`t10.2`) **and** by a round-trip parser test against `Model`.

**`t11` Project A in POMDP terms (B).** Express §5.1 as a policy π(b) on the same belief space.
- The localization phase is a one-step lookahead with a surrogate reward −H(b) (an information reward, not the task reward).
- Travel is the QMDP-like "most likely state" policy (Cassandra et al. 1996 call it MLS).
- Declaration is a belief threshold.

Name the approximations: myopic horizon, entropy surrogate, MLS travel that ignores uncertainty, hand-tuned thresholds, and no collision-risk term in the original Project A code. These are exactly what the POMDP solver removes, and the report should say so.

### Phase p3: Solver (wk 6–10)

**`t12` SARSOP on 5×5 (A).**
- Use the APPL toolkit (`pomdpsol` binary, C++; build it on WSL or Linux), or `NativeSARSOP.jl`, or the `pomdp_py` SARSOP wrapper.
- Run `pomdpsol model.pomdp --precision 0.01 --timeout 3600 -o policy.policy`.
- Log the upper/lower bound gap over time (`t12.0`: gap < 1%), wall time and memory (`t12.1`).
- `scripts/solve_sarsop.sh maze λ` does generate → solve → copy the policy and its log (`t12.2`).

**`t13` alpha-vector policy (S).**
- Parse `.policy` (XML `<Vector action="a" obsValue="...">`) into `Γ ∈ ℝ^{K×(N+1)}` and `act[K]`.
- `AlphaPolicy.act`: update the belief (§3.4), compute `a = act[argmax_k Γ_k·b]`.
- Run it through the unchanged `runner.py` (`t13.1`).
- Sanity checks: the empirical mean of `−(T + λ·fail)` must match SARSOP's bounds at b₀ within the CI (the γ<1 bias is small), and the policy should never `DECLARE` while `b(G)` is low.

**`t14` λ sweep (B).**
- λ log-spaced in [5, 5000] (≥ 8 values, `t14.0`). Add λ values adaptively where neighbouring points are far apart.
- Plot on the same axes as `t8` (`t14.1`).
- Compute the lower convex hull explicitly and state that λ-optimal policies lie on it (`t14.2`). A target d between two hull vertices is reached by **randomizing** between the two adjacent policies at episode start.

**`t15` scaling (A).** In this order:
1. **Observation compression:** merge observations whose `Z[:, o]` columns are proportional (lossless), and drop `o` with `max_s Z[s,o] < 1e-6`.
2. **Reduce R or the number of rays** for 16×16 if needed. Document the information loss using the ambiguity score.
3. **Faster convergence:** tune SARSOP (`--randomization`, a fixed time budget) and accept a known bound gap.
4. **Online fallback:** POMCP (Silver & Veness 2010) or DESPOT (Somani et al. 2013) on the exact `Model` as a generative model, with λ in the reward. Use QMDP as the rollout policy. Report the per-step planning time.
5. **Bounds as baselines:** QMDP (an upper bound on the value) and FIB. Cheap, and they add points to the comparison.

Targets: `t15.1` requires 10×10 in under 12 h. For 16×16, write the decision document (`t15.3`).

### Phase p4: Comparison and submission (wk 10–13)

**`t17` head-to-head (B).**
- ≥ 1,000 runs per point, on the same mazes and seeds (paired).
- Primary metric: **E[T] at fixed d ∈ {1%, 5%, 10%}**, by linear interpolation along each algorithm's lower hull with bootstrap CIs.
- Significance: a paired bootstrap or permutation test on T, plus a two-proportion z-test or Fisher's exact test on failure rates.
- Sample size: resolving a 1% failure rate to ±0.6% (95%) needs about 1,000 runs. For d = 1% claims, prefer ≥ 3,000 runs.

**`t18` sensitivity (A).** Re-solve or re-run while varying R ∈ {1, 2, 3}, `p_miss` ∈ {0, 0.05, 0.1, 0.2} and `p_steps`/`p_drift` from easy to hard. Also run a **model-mismatch** test: the solver assumes the nominal model while the environment uses a perturbed one. This is the controlled version of the MuJoCo gap.

**`m4` MuJoCo evaluation (S).** Both policies run on `MujocoEnv` with the **calibrated** `Model`. ≥ 200 runs × 3 values of d × 2 algorithms. Report the sim-to-sim gap table (`m4.2`) and the `n_fallbacks` count (impossible observations), which measures model mismatch. Videos (`m4.3`): show the POMDP policy keeping its distance from walls when collision risk is high. This is the "coastal" behavior (Roy & Thrun 1999) and should appear naturally under a large λ.

---

## 5. Algorithms in detail

### 5.1 Ported Project A heuristic (`policies/projecta.py`, `t7`)

Belief b over free cells, updated with §3.4.

```
act(o):
  b ← update(b, a_prev, o)
  if b[G].sum() ≥ τ_declare:            return DECLARE                  # declaration rule
  localized = P_max(b) ≥ τ_loc and P_runner_up_distant(b)/P_max(b) < ρ_amb
  safe(a)   = Σ_s b(s)·C[a,s] ≤ ε_col                                      # [NEW vs Project A] collision gate
  if not localized:                                                        # LOCALIZING
      return argmax_{a ∈ moves, safe(a)}  H(b) − Σ_o P(o|b,a)·H(b^{a,o}) − move_cost
  else:                                                                    # TRAVELING
      ŝ = argmax b ;  path = BFS_8conn(ŝ → G) ;  a = first step of path
      return a if safe(a) else (safest move that does not increase dist-to-goal, else localize step)
```

How each line maps to Project A's code, and what changes:

| Project A (`lidar_sim`) | Project B port |
|---|---|
| Belief `(r, c, 8)`; `update_probability` / `predict_forward` / `predict_turn` as Python loops | Belief `(N,)` over free cells; §3.4 in numpy |
| Likelihood floor `1e-9` per state + `_normalize_or_uniform` | Exact zeros allowed (discrete Z); uniform-mixing fallback only on an impossible observation, counted in the CSV |
| `select_best_action`: 7 rotations + 4 "turn then step" moves; `move_cost = 0.10`, `turn_cost = 0.02` per 45° | 8 absolute moves, no rotations; `move_cost` kept (default 0.10 nats); diagonal moves cost the same |
| `E[H]` over the top-16 hypotheses' noiseless readings (`prob_floor = 1e-4`) | Exact sum over the finite O (below); top-K kept as an option for speed on 16×16 |
| `dominant_position_modes(min_separation=2)`, `LOCALIZATION_THRESHOLD = 0.25`, `LOCALIZATION_AMBIGUITY_RATIO = 0.5` | Same rule on the 2-D belief: `τ_loc = 0.25`, `ρ_amb = 0.5`, separation = Chebyshev ≥ 2 |
| `IS_LOST_THRESHOLD = 0.15` while MOVING; `reset_auto` keeps the belief | `τ_lost = 0.15` hysteresis; the belief is never reset |
| A\* over `(r, c, θ)` (forward 1, 45° turn 0.25, 180° turn 0.5), Manhattan heuristic, replan when off-path | BFS or Dijkstra over cells with 8-connectivity under D2 (no heading); replan every step from the MAP cell |
| `DONE` when the **true** pose equals the goal | `DECLARE` when `b(G) ≥ τ_declare` (a belief decision) |
| Wall blocks the move silently | `C[a, s]` collision risk, gated by `ε_col` |

- "Distant runner-up": the highest-probability cell at Chebyshev distance ≥ 2 from the arg-max, exactly as in `dominant_position_modes`.
- **Exact expectation:** Project A approximated E_z using representative noiseless readings. Here O is finite and small, so compute it exactly in vectorized form: `P(o|b,a) = (b̄ᵀ Z)[o]`, and the posteriors are `Z[:,o] ⊙ b̄`, all at once as an `(N, |O|)` matrix. Ignore observations with probability below 1e-9.
- The state machine collapses to `LOCALIZING ↔ TRAVELING → DECLARE`. Going back to localizing is automatic when `localized` turns false (Project A's `lost` threshold becomes hysteresis: `τ_lost < τ_loc`).
- Knobs that trace the Pareto curve: `τ_declare` (main), `ε_col`, `τ_loc`, `ρ_amb`.

### 5.2 QMDP baseline [NEW, cheap and useful]

Solve the fully observable MDP on `(T, C, R_λ)` by value iteration to get `Q(s,a)`. Then `π(b) = argmax_a Σ_s b(s)Q(s,a)`. Known flaw: it never takes pure information-gathering actions, but here moving also senses. It is about 50 lines, gives one more λ-tunable curve, and acts as the POMCP rollout policy.

### 5.3 Pareto analysis (`analysis/pareto.py`)

- Point estimate per configuration: `(f̂, T̄)` with a Wilson CI on f and a bootstrap CI on T.
- Non-dominated set; lower convex hull (Andrew's monotone chain on points sorted by f).
- `T_at(d)`: interpolation on the hull, with a CI by bootstrapping runs inside each configuration.
- Always plot both algorithms on the same axes per maze size, with a log scale on f when d = 1% matters.

---

## 6. Risks and mitigations

| Risk | Signal | Mitigation |
|---|---|---|
| Porting Project A changes its behavior | The ported heuristic does clearly better or worse than the original in the same setting | Run the original `lidar_sim` and the port side by side with `wall_mode: stop`, 4 directions, the `tof5` sensor and the same seeds on a 15×20 Project A map. Fix any difference before running `t8` |
| Project A defaults are nearly deterministic (`p_steps` ≈ (0.02, 0.97, 0.01)) | Every algorithm gets near-zero failure, so the frontier collapses | Evaluate at both the Project A noise level and a harder one. Choose the main setting after `m3` calibration |
| SARSOP does not converge for 10×10 or 16×16 | Gap > 5% after 12 h | §4 `t15` steps 1–4. Report the bounds honestly. POMCP for the large size |
| Observation space blow-up | \|O\| > ~2,000 | D5 with 4 rays, R ≤ 2, lossless merging |
| λ sweep leaves gaps on the frontier | Big jumps in f between adjacent λ | Adaptive λ bisection plus policy mixing. Mention CPBVI/CALP as future work |
| Discount bias (γ < 1 against undiscounted T) | SARSOP value ≠ empirical cost | γ = 0.999. Report the gap. Episodes are short (T ≲ 100) |
| MuJoCo cannot reproduce discrete 0/1/2 steps | Calibrated `p_steps` has mass outside {0,1,2} or between cells | Controller with explicit target cells. Snap with a tolerance. Report "between cells" as a separate outcome |
| Holonomic robot rejected by the supervisor | `t1.1` | Differential drive with in-place turns that are excluded from T. Same discrete interface |
| Python env too slow | `t5.2` fails | Remove prints, vectorize, use a process pool. N ≤ 256 keeps everything tiny |

## 7. Testing checklist (CI)

- `test_model.py`: stochasticity (`T+C=1`, `Z` rows sum to 1); a deterministic special case; diagonal rule D2 on hand-built 2×2 mazes; mirror and rotation symmetry (`np.rot90` on `wall_map` with the matching channel permutation; T and Z must rotate the same way).
- `test_env.py`: empirical frequencies against `Model` (χ²); seeds reproducible; collision terminates.
- `test_belief.py`: normalized; true state never gets zero probability (property test); fallback triggers only on impossible observations.
- `test_pomdp_io.py`: round trip `Model → .pomdp → parse → arrays` with equal values. The Tiger file loads and the solved Tiger policy matches the known optimal behavior.
- `test_policies.py`: Project A at 90% success or better on 5×5 with defaults (`t7.1`). The alpha policy never declares when `b(G) < 0.5`. Project A's `active_localization._demo()` checks are ported as tests.
- `test_noise.py`: Project A's `test_discrete_noise.py` ported as is (probabilities sum to 1; sampled frequencies match).
- `test_port_parity.py`: with `wall_mode: stop`, 4 directions and `tof5`, the ported filter gives the same posterior as Project A's `update_probability` (headings summed out) on random beliefs.
- `test_mjcf.py`: the wall-geom count equals the unique wall edges. The rangefinder reading at a known cell matches the expected distance ±5 mm with noise off.

## 8. Glossary (consistent with Project A)

- **Belief** `b`: distribution over free cells.
- **Prediction step / correction step**: motion update and sensor update.
- **Kidnapped**: start pose unknown, so b₀ is uniform.
- **Active localization**: choosing actions for information.
- **Declare**: the terminal action claiming arrival.
- **Failure**: `COLLISION`, `WRONG_DECLARATION` or `TIMEOUT`.
- **d**: allowed failure probability. **λ**: Lagrange penalty on failure.
- **Frontier**: the set of non-dominated (failure rate, E[T]) points. **Hull**: its lower convex envelope.
