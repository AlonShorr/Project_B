# Kidnapped Robot Project (Project B): Roadmap

> **For LLM assistants:** This file is the main context document for Project B. Read it before you plan, write code, or write report text for this project. It merges two sources:
> 1. The supervisor's roadmap: `Kidnapped Robot Project - roadmap.docx` (problem definition and the four phases).
> 2. The team's detailed 13-week plan: `gantt-project-b/public/plan.json` (tasks, owners, weeks and acceptance criteria). It is shown as a Hebrew Gantt chart.
>
> For the engineering design (models, interfaces, math, Project A code reuse, tests), read `ROADMAP_TECHNICAL.md`.
>
> If this file and `plan.json` disagree on schedule or criteria, `plan.json` wins. Task IDs (`t3`, `m1`, …) and criterion IDs (`t3.2`) here match `plan.json`. Use them when you refer to work items.

---

## 1. Team and context

| Code | Person | Main areas |
|------|--------|------------|
| `S`  | Segev  | Simulators (Python and MuJoCo), experiment infrastructure, evaluation |
| `A`  | Alon   | Sensing model, Project A algorithm port, POMDP formulation, solver |
| `B`  | Both   | Shared tasks (learning, baselines, comparison, report) |

- **Project A code** (previous semester) is the Git repo `AlonShorr/Project_A` (Segev's clone: `\\wsl.localhost\Ubuntu\home\logan\Project_A`):
  - Project A's simulator and algorithms: `development/SA_Simulator/lidar_sim/` (Bayes filter, active localization, A\*, map generator, Pygame GUI). `ROADMAP_TECHNICAL.md` §0 describes it in detail.
  - ESP32 localization firmware: `development/localization/` (reference only)
  - Predecessor team's simulator and firmware: `legacy/micromouse-py-sim-main/`, `legacy/micromouse-robot-main/`
- **Project A documents** are in `C:\Users\segev\Documents\Technion\Semester_7\ProjectA`: the final report (`Micromouse_Active_Localization_Final_Project_Report.docx`) and the background paper *Active Localization for Bayesian-Grid Micromouse: Entropy-Minimizing Move Selection*.
- **Change from the supervisor's plan:** the experiment on the physical micro-mouse robot is replaced by a **MuJoCo simulation** with continuous physics and realistic uncertainty. This still needs the supervisor's approval (`t1.1`).

---

## 2. Problem definition

- **Setting:** a 2D grid maze with obstacles. A robot must reach a goal cell as fast as possible while meeting a required success probability. Its start position is unknown ("kidnapped").
- **Actions:** move in 8 directions (U, D, L, R and the 4 diagonals), plus a **declare "arrived"** action.
- **Sensing:** the robot senses obstacles around it within range **R** (a system parameter). It does not need to rotate.
- **Noise:**
  - *Motion:* a move command can advance 0, 1 or 2 cells in the chosen direction, or drift to an adjacent direction, according to a given distribution.
  - *Sensing:* an obstacle can be **missed** with some probability.
- **Run end:** the robot declares "arrived" when it believes it is at the goal. Run duration **T** is the number of steps until then.
- **Failure** happens when either:
  1. the robot declares "arrived" but is not at the goal, or
  2. the robot collides with an obstacle (the run stops immediately).
- **Objective:** minimize **E[T]** subject to **P(failure) ≤ d**.
- **Main output:** a **Pareto frontier** of mean T against failure rate, for each algorithm, on mazes of several sizes.

## 3. Research goal

Develop an algorithm based on a **POMDP** (Partially Observable Markov Decision Process) solver for this problem. Compare it with the Project A heuristic, both in simulation and in a "real-world-like" experiment (MuJoCo).

**Project A baseline algorithm:** it alternates between a localization phase and a goal-travel phase.
- Localization steps are chosen greedily to minimize the entropy of the location belief.
- Travel follows an A\* path to the goal.
- A switching criterion decides when to move from localization to travel. It is still undefined (the "?" in the supervisor's document) and is settled in `t1.2`.

**Planned POMDP approach:** add a failure penalty **λ** to the reward. Solve with a point-based solver (for example SARSOP) instead of plain value iteration. Sweep λ to trace the Pareto frontier.
- Note: a λ sweep only recovers the **convex hull** of the frontier (`t14.2`).
- To scale up: shrink the observation space, use point-based methods, and fall back to POMCP if needed.

---

## 4. Timeline overview (13 weeks)

| Phase | ID | Weeks | Summary |
|-------|----|-------|---------|
| Kickoff and learning | `p0` | 1–3 | Agree scope with the supervisor and build shared POMDP knowledge |
| Phase 1: Simulator and baseline | `p1` | 1–5 | Adapt the Python simulator to the new definition. Produce the Project A Pareto curve |
| MuJoCo environment | `pm` | 3–9 | Same robot and interface as the Python sim, with continuous physics and real-world uncertainty |
| Phase 2: POMDP formulation | `p2` | 4–6 | Express the problem and the Project A algorithm in POMDP terms |
| Phase 3: Value-iteration-based solver | `p3` | 6–10 | Optimal solver with λ penalty, then scaling. This is the main research part |
| Phase 4: Comparison and submission | `p4` | 10–13 | Head-to-head comparison in both simulators, report and presentation |

### Milestones

| Week | Milestone |
|------|-----------|
| 1  | Scope approved by the supervisor |
| 5  | Baseline curve ready, MuJoCo environment running, first review meeting |
| 6  | POMDP formulation ready |
| 9  | Noise model calibrated from MuJoCo |
| 10 | POMDP curves for all maze sizes |
| 12 | Full draft sent to the supervisor |
| 13 | Submission |

---

## 5. Detailed tasks and acceptance criteria

Format: **`id` · title** (owner, weeks). Each criterion is listed as `id.n`.

### Phase p0: Kickoff and learning

**`t1` · Agree scope with the supervisor** (B, wk 1). Get written agreement on what is required before investing time.
- `t1.0` Email sent to the supervisor by Tuesday of week 1
- `t1.1` Supervisor approved MuJoCo as a replacement for the physical robot experiment
- `t1.2` Localization-to-travel switching criterion defined
- `t1.3` Default values set for d, R and the noise parameters
- `t1.4` Two review meetings scheduled: end of week 5 and end of week 11

**`t2` · Learn POMDPs** (B, wk 1–3). Shared theory for Phases 2 and 3.
- `t2.0` Both read the tutorial at https://www.pomdp.org/tutorial/index.html
- `t2.1` Both read sections 1–4 of Kaelbling, Littman & Cassandra (1998): https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf
- `t2.2` Each solved two value-iteration steps by hand on the Tiger problem
- `t2.3` One-page summary covering MDPs, belief, value iteration and policy iteration

Topics to cover: MDPs, states, actions, observations, reward function, value iteration, policy iteration.

### Phase p1: Simulator and baseline

**`t3` · New motion model** (S, wk 1–2). 8 directions. Each step yields 0, 1 or 2 cells or drifts to an adjacent direction. A collision ends the run.
- `t3.0` 8 move actions and the declare-arrival action implemented
- `t3.1` Unit test: transition probabilities sum to 1 for every (state, action)
- `t3.2` Over 10,000 samples, the empirical distribution is within ±1% of the model
- `t3.3` A collision ends the run and is logged as its own failure type

**`t4` · New sensing model** (A, wk 1–2). Obstacle sensing within range R, no rotation, with a miss probability.
- `t4.0` Observation model O(o|s) and world model written in the formalization document
- `t4.1` Unit test: the belief stays normalized after every update
- `t4.2` Unit test: with no misses, one observation distinguishes cells with different signatures

**`t5` · Central config and headless runs** (S, wk 2–3). Every comparison in the project depends on this.
- `t5.0` All parameters (noise, R, thresholds, costs, seed) in a single config file
- `t5.1` Two runs with the same seed give identical results
- `t5.2` 1,000 runs on a 10×10 maze take under 10 minutes
- `t5.3` CSV output per run: T, end type, seed and parameters

**`t6` · Maze suite in three sizes** (S, wk 3–4). Small 5×5, medium 10×10, large 16×16 (competition size).
- `t6.0` At least 3 mazes per size
- `t6.1` Ambiguity score computed per maze (how many cells share a sensing signature)
- `t6.2` Mazes stored in the repo and loaded by name

**`t7` · Port the Project A algorithm** (A, wk 2–4). Greedy entropy-based localization, A\* travel, and arrival declaration.
- `t7.0` Switching criterion and declaration threshold exposed as parameters
- `t7.1` At least 90% success on a 5×5 maze with default parameters
- `t7.2` Runs through the experiment runner with no code changes between mazes

**`t8` · Project A Pareto curve** (B, wk 4–5). The baseline the new solver has to beat.
- `t8.0` At least 20 parameter configurations, 500 runs each, on every maze
- `t8.1` Plot of mean T against failure rate for each maze size, with 95% confidence intervals
- `t8.2` Results sent to the supervisor before the review meeting

**Phase 1 outcome:** a Pareto frontier of time against failure rate for the Project A algorithm, on small, medium and large mazes.

### Phase pm: MuJoCo environment (replaces the physical experiment)

**`m1` · MuJoCo maze and robot** (S, wk 3–5). Same robot as the Python simulator, with a low-level controller that turns each discrete action into continuous motion.
- `m1.0` MJCF generator that reads the same maze file as the Python simulator
- `m1.1` Robot geometry, motion directions and sensing range R match the Python simulator
- `m1.2` Low-level controller executes each action and returns the cell where the robot stopped, or a collision
- `m1.3` Shared interface: the same policy code runs on both simulators unchanged
- `m1.4` 100 headless runs on a 5×5 maze in under 10 minutes

**`m2` · Uncertainty sources in MuJoCo** (S, wk 6–7). The uncertainties a real robot would face, each with its own parameter.
- `m2.0` Wheel slip, and friction that varies between runs
- `m2.1` Controller errors: noise in velocity commands and stopping time
- `m2.2` Distance-dependent sensor noise, and readings that miss a wall
- `m2.3` Every source can be switched on or off in the config
- `m2.4` Parameter table with a source for every value (datasheet or literature)

**`m3` · Calibrate the noise model from MuJoCo** (S, wk 8–9). Measure in MuJoCo the probabilities that the Python simulator and the POMDP assume.
- `m3.0` At least 1,000 commands per action type, logging the cell where the robot stopped
- `m3.1` Probabilities of 0, 1 or 2 steps and of adjacent-direction drift estimated with confidence intervals
- `m3.2` Sensing miss probability estimated the same way
- `m3.3` Python simulator and POMDP file updated to the calibrated values

### Phase p2: POMDP formulation

**`t10` · POMDP formulation of the problem** (A, wk 4–6). States, actions, observations, transitions, reward, and terminal states (collision and declaration).
- `t10.0` Written chapter with S, A, O, T, Z and R, including the failure penalty λ
- `t10.1` POMDP file generator for a 5×5 maze
- `t10.2` The file loads into the solver without errors

**`t11` · Project A in POMDP terms** (B, wk 5–6). The Phase 2 deliverable in the supervisor's document.
- `t11.0` Pseudocode that explicitly defines actions, observations and the underlying MDP
- `t11.1` Written explanation of where the algorithm approximates: greediness, one-step lookahead, hand-tuned thresholds

### Phase p3: Value-iteration-based solver (main research)

**`t12` · Solver on a small maze** (A, wk 6–8). A point-based solver (for example SARSOP) instead of full value iteration.
- `t12.0` Solver converges on 5×5 with an upper/lower bound gap under 1%
- `t12.1` Runtime and memory documented
- `t12.2` A single script reproduces the solution from scratch

**`t13` · Run the policy in the simulator** (S, wk 7–8). The solver's policy runs through the same experiment runner as Project A.
- `t13.0` Load α-vectors and choose the action from the current belief
- `t13.1` 500 runs on 5×5 pass with no change to the experiment runner

**`t14` · λ sweep and POMDP Pareto curve** (B, wk 8–9). Each λ value gives a different point on the curve.
- `t14.0` At least 8 λ values on a 5×5 maze
- `t14.1` Curve plotted on the same graph as the Project A curve
- `t14.2` Documented that a λ sweep finds only the convex hull of the frontier

**`t15` · Scale to medium and large mazes** (A, wk 9–10). Reduce the observation space, use point-based methods, and fall back to POMCP if needed.
- `t15.0` Solver runs with the parameters calibrated from MuJoCo
- `t15.1` Solution for a 10×10 maze in under 12 compute-hours
- `t15.2` At least one documented improvement, with time and quality compared against the baseline
- `t15.3` Written decision for 16×16: solve, approximate, or explain why not

**Phase 3 outcome:** a new algorithm that is optimal (possibly slow) and tunable, where d can be varied to trace the Pareto frontier.

### Phase p4: Comparison and submission

**`m4` · Evaluation in MuJoCo** (S, wk 10–12). Shows the policy still works when the world is not discrete. Replaces the physical experiment.
- `m4.0` Both algorithms run in MuJoCo through the shared interface
- `m4.1` At least 200 runs per algorithm for d = 1%, 5% and 10%
- `m4.2` Sim-to-sim gap: T and failure rate in MuJoCo against Python, with confidence intervals
- `m4.3` Two demo videos: a successful run, and a run where the solver keeps away from a dangerous wall

**`t17` · Head-to-head comparison** (B, wk 10–11). The main result of the project.
- `t17.0` Both curves on the same mazes, at least 1,000 runs per point
- `t17.1` Table of each algorithm's mean T for d = 1%, 5% and 10%
- `t17.2` Statistical significance test for the differences

**`t18` · Sensitivity analysis** (A, wk 11). How the results change when the model changes.
- `t18.0` Sensitivity plot for sensing range R
- `t18.1` Sensitivity plot for miss probability
- `t18.2` Sensitivity plot for motion noise

**`t19` · Report and presentation** (B, wk 11–13). Alon writes the POMDP and solver chapters. Segev writes the two simulators and the evaluation chapters. Week 13 doubles as a buffer.
- `t19.0` Report outline and chapter split agreed by the end of week 11
- `t19.1` Full draft sent to the supervisor by the end of week 12
- `t19.2` 15-minute presentation, rehearsed at least once
- `t19.3` Final submission by the end of week 13

---

## 6. Dependency map

```
t1 (scope) ─┬─> t3, t4 (motion/sensing models) ─> t5 (config + headless runner) ─> t6 (mazes)
            │                                                    │
            │                                       t7 (Project A port) ─> t8 (baseline Pareto)
            │
            ├─> m1 (MuJoCo env) ─> m2 (uncertainties) ─> m3 (calibration) ─> t15, m4
            │
            └─> t2 (POMDP learning) ─> t10 (formulation) ─> t11 (Project A as POMDP)
                                            │
                                            └─> t12 (solver 5×5) ─> t13 (policy in sim) ─> t14 (λ sweep)
                                                                                              │
                                                     t15 (scaling) ─> t17 (head-to-head) <────┘
                                                                         ├─> t18 (sensitivity)
                                                     m4 (MuJoCo eval) ───┴─> t19 (report)
```

## 7. Engineering conventions for assistants

- **One interface, two simulators.** Policy code (Project A heuristic and POMDP policy) must run unchanged on the Python simulator and on MuJoCo (`m1.3`, `t13.1`, `m4.0`).
- **One config file** holds every tunable parameter: noise, R, thresholds, costs, λ, d and seed (`t5.0`). Never hard-code parameters.
- **Reproducibility:** seeded runs must be deterministic (`t5.1`). The solver must be reproducible from one script (`t12.2`).
- **Output:** one CSV row per run with T, end type (`success`, `wrong_declaration`, `collision`), seed and parameters (`t5.3`).
- **Statistics:** report 95% confidence intervals on every curve, and test significance for head-to-head claims.
- **Maze sizes:** 5×5 (small), 10×10 (medium), 16×16 (large, competition size). Mazes are loaded by name from the repo.
- **Performance targets:** 1,000 Python runs on 10×10 in under 10 min; 100 MuJoCo runs on 5×5 in under 10 min; POMDP solve on 10×10 in under 12 compute-hours.

## 8. Open decisions

- Switching criterion between localization and travel in the Project A algorithm (`t1.2`).
- Default values of d, R and the noise parameters (`t1.3`).
- Supervisor approval of MuJoCo instead of the physical robot (`t1.1`).
- Choice of solver (SARSOP or another point-based method, with POMCP as fallback) and the scaling strategy for 16×16 (`t15.3`).

## 9. Related files

- `Kidnapped Robot Project - roadmap.docx`: the supervisor's original roadmap
- `gantt-project-b/`: shared Gantt and progress tracker (Vercel + Upstash Redis). `public/plan.json` holds the plan. See its README for the rules on editing IDs.
- `gantt_project_b.html`: standalone copy of the Gantt page
