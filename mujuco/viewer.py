"""Watch and drive the robot in the MuJoCo viewer. Run from the mujuco folder:

    ..\\.venv\\Scripts\\python viewer.py                                  # drive it yourself (small/m5_a)
    ..\\.venv\\Scripts\\python viewer.py --maze medium/m10_a --start 0,0
    ..\\.venv\\Scripts\\python viewer.py --policy projecta --seed 3       # watch the algorithm
    ..\\.venv\\Scripts\\python viewer.py --policy projecta --episodes 10 --speed 3
    ..\\.venv\\Scripts\\python viewer.py --mj-set mecanum.controller.yaw_mode=held --mj-set mecanum.noise.imu_on=true

Every action goes through MujocoGridEnv, the same env the batch runner uses, so
outcomes (collision, wrong declaration, success) and observations are exactly
what an experiment would record. The robot's belief (a Bayes filter over cells,
the policy's own belief in --policy mode) is drawn on the floor as a heat map.

Keys (also printed in the viewer):
    arrows / numpad 8 6 2 4      move N E S W one cell
    Home PgUp End PgDn / 7 9 1 3 move NW NE SW SE
    Enter / numpad 5             DECLARE "arrived"
    Insert                       new episode (next seed)
    Delete                       show / hide the belief heat map
    numpad + / -                 faster / slower (policy mode)
    Pause or numpad 0            pause / resume the policy
"""

from __future__ import annotations

import argparse
import collections
import functools
import sys
import time

import mujoco
import numpy as np

import sim  # noqa: F401  (puts lidar_sim on sys.path)
from sim.mj_config import load_mj_config
from sim.mj_grid_env import make_mujoco_env
from sim.mjcf import CELL
from belief import BayesFilter, uniform_belief
from motion_model import Action, Outcome
from policies import EnvInfo, make_policy
from projecta import Model
from runner import LoadedMaze, policy_rng, resolve_maze
from sensing import decode_observation
from sim_config import load_config

# GLFW key codes
KEYS = {
    265: Action.N, 262: Action.E, 264: Action.S, 263: Action.W,            # arrows
    268: Action.NW, 266: Action.NE, 269: Action.SW, 267: Action.SE,        # Home PgUp End PgDn
    328: Action.N, 329: Action.NE, 326: Action.E, 323: Action.SE,          # numpad
    322: Action.S, 321: Action.SW, 324: Action.W, 327: Action.NW,
    257: Action.DECLARE, 335: Action.DECLARE, 325: Action.DECLARE,         # Enter, KP Enter, KP 5
}
KEY_NEW, KEY_BELIEF, KEY_FASTER, KEY_SLOWER = 260, 261, 334, 333           # Insert, Delete, KP +, KP -
KEY_PAUSE = (284, 320)                                                     # Pause, KP 0
HELP = ("arrows / numpad: N E S W\nHome PgUp End PgDn: NW NE SW SE\nEnter / numpad 5: DECLARE\n"
        "Insert: new episode   Delete: belief on/off\nnumpad +/-: speed   Pause / numpad 0: pause policy")


class ViewerApp:
    """Episode logic and drawing, independent of the window (testable with a fake handle)."""

    def __init__(self, cfg, maze: LoadedMaze, mj_cfg: dict, policy_name: str | None = None,
                 seed: int = 0, start: tuple[int, int] | None = None, speed: float = 1.0, fps: float = 60.0):
        self.cfg, self.maze, self.mj_cfg = cfg, maze, mj_cfg
        self.env = make_mujoco_env(cfg, maze, mj_cfg)
        self.sim = self.env.sim
        self.seed, self.start, self.speed, self.fps = seed, start, speed, fps
        self.policy = None
        if policy_name:
            pcfg = cfg.replace([f"policy.name={policy_name}"])
            self.policy = make_policy(pcfg.policy.name, pcfg.policy.selected_params, cfg=pcfg)
        # Belief for manual mode (and for policies without their own): exact Bayes filter.
        m = Model.build(maze.wall_map, maze.goal_cells, self.env.params, self.env.sensing, )
        self.filter = BayesFilter(m.T, m.C, m.Z)
        self.b0 = uniform_belief(maze.rows, maze.cols, exclude=maze.goal_cells)
        self.show_belief = True
        self.paused = False
        self.keys: collections.deque[int] = collections.deque()
        self.handle = None
        self.status = ""
        self._last_sync_sim = 0.0
        self._wall0 = self._sim0 = 0.0

    # ---------- episodes ----------
    def new_episode(self) -> None:
        res = self.env.reset(self.seed, start=self.start)
        self.obs, self.info, self.last_action = res.obs, res.info, None
        self.filter.reset(self.b0, self.obs)
        if self.policy is not None:
            self.policy.reset(EnvInfo.from_env(self.env), policy_rng(self.seed))
        self.status = f"episode seed {self.seed}: kidnapped"
        self._wall0, self._sim0 = time.perf_counter(), self.sim.data.time

    def act(self, action: Action) -> None:
        if self.env.done:
            return
        self._wall0, self._sim0 = time.perf_counter(), self.sim.data.time
        res = self.env.step(action)
        self.last_action = action
        self.obs, self.info = res.obs, res.info
        if action.is_move and not self.env.done:
            self.filter.update(action, self.obs)
        out = self.env.log.outcome
        self.status = f"{action.name} -> {out.value}" if out is not Outcome.RUNNING else f"{action.name}"

    def policy_step(self) -> None:
        if self.policy is None or self.env.done or self.paused:
            return
        info = self.info if self.policy.privileged else {"t": self.env.log.steps, "last_action": self.last_action}
        self.act(self.policy.act(self.obs, info))

    def belief(self) -> np.ndarray:
        agent = getattr(self.policy, "agent", None)
        if agent is not None and getattr(agent, "belief", None) is not None:
            return np.asarray(agent.belief)
        return self.filter.belief

    # ---------- input ----------
    def key_callback(self, keycode: int) -> None:          # viewer thread: only queue
        self.keys.append(keycode)

    def handle_keys(self) -> None:
        while self.keys:
            k = self.keys.popleft()
            if k == KEY_NEW:
                self.seed += 1
                self.start = None
                self.new_episode()
            elif k == KEY_BELIEF:
                self.show_belief = not self.show_belief
            elif k == KEY_FASTER:
                self.speed = min(self.speed * 2, 32.0)
            elif k == KEY_SLOWER:
                self.speed = max(self.speed / 2, 0.125)
            elif k in KEY_PAUSE:
                self.paused = not self.paused
            elif k in KEYS and self.policy is None:
                self.act(KEYS[k])

    # ---------- drawing ----------
    def on_step(self, data: mujoco.MjData) -> None:
        """Called after every physics step: show a frame every 1/fps of simulated time and
        keep the playback at ``speed`` x real time."""
        if 0.0 <= data.time - self._last_sync_sim < self.speed / self.fps:
            return                                         # (time restarts at 0 on a new episode)
        self._last_sync_sim = data.time
        self.draw()
        lag = (data.time - self._sim0) / self.speed - (time.perf_counter() - self._wall0)
        if lag > 0:
            time.sleep(lag)

    def texts(self) -> list[tuple]:
        env, b = self.env, self.belief()
        digits = decode_observation(self.obs, env.sensing) if self.obs is not None else None
        R = env.sensing.R
        obs_txt = " ".join(f"{d}={'far' if v == R + 1 else v}" for d, v in zip("NESW", digits)) if digits else "-"
        rmap, cmap = divmod(int(np.argmax(b)), self.maze.cols)
        mode = f"policy {self.policy.name}" + (" (paused)" if self.paused else "") if self.policy else "manual"
        left = "\n".join(["mode", "maze", "robot", "true cell", "step", "observation", "belief max", "P(goal)", "status"])
        right = "\n".join([
            f"{mode}   speed x{self.speed:g}",
            f"{self.maze.name} ({self.maze.rows}x{self.maze.cols})",
            f"{self.mj_cfg['robot']}" + (f"  yaw {np.degrees(self.sim.yaw()):+.2f} deg" if hasattr(self.sim, "yaw") else ""),
            f"{env.cell}",
            f"{env.log.steps} / {env.max_steps}",
            obs_txt,
            f"({rmap}, {cmap})  p = {b.max():.2f}",
            f"{b[[r * self.maze.cols + c for r, c in self.maze.goal_cells]].sum():.3f}",
            self.status,
        ])
        return [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, HELP, None)]

    def overlay(self, scn: mujoco.MjvScene) -> None:
        """Goal cells (green) and the belief heat map (orange, alpha ~ probability) on the floor."""
        scn.ngeom = 0
        eye = np.eye(3).ravel()

        def add(r, c, rgba, z, half):
            if scn.ngeom >= scn.maxgeom:
                return
            x, y = (c + 0.5) * CELL, -(r + 0.5) * CELL
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([half, half, 0.0005]), np.array([x, y, z]), eye, np.asarray(rgba, np.float32))
            scn.ngeom += 1

        for r, c in self.maze.goal_cells:
            add(r, c, (0.1, 0.8, 0.2, 0.5), 0.0006, CELL * 0.47)
        if self.show_belief:
            b = self.belief()
            top = max(float(b.max()), 1e-9)
            for i in np.flatnonzero(b > 1e-3):
                r, c = divmod(int(i), self.maze.cols)
                add(r, c, (1.0, 0.45, 0.0, 0.15 + 0.75 * float(b[i]) / top), 0.0012, CELL * 0.40)

    def draw(self) -> None:
        h = self.handle
        if h is None:
            return
        with h.lock():
            self.overlay(h.user_scn)
        h.set_texts(self.texts())
        h.sync()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--maze", default=None, help="maze name (default: the config's maze.name)")
    p.add_argument("--start", default=None, help="start cell r,c (default: random, as in experiments)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", default=None, help="watch this policy (e.g. projecta, oracle_bfs, random)")
    p.add_argument("--episodes", type=int, default=1, help="policy mode: episodes to play back to back")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (x real time)")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="experiment config override")
    p.add_argument("--mj-set", dest="mj_overrides", action="append", default=[], help="MuJoCo config override")
    p.add_argument("--exit-after", type=float, default=None, help=argparse.SUPPRESS)   # smoke tests
    args = p.parse_args(argv)

    cfg = load_config(overrides=args.overrides + ([f"maze.name={args.maze}"] if args.maze else []))
    mj_cfg = load_mj_config(overrides=args.mj_overrides)
    start = tuple(int(v) for v in args.start.split(",")) if args.start else None
    app = ViewerApp(cfg, resolve_maze(cfg), mj_cfg, args.policy, args.seed, start, args.speed)
    app.sim.on_step = app.on_step

    import mujoco.viewer

    t_open = time.perf_counter()
    with mujoco.viewer.launch_passive(app.sim.model, app.sim.data, key_callback=app.key_callback) as v:
        app.handle = v
        rows, cols = app.maze.rows, app.maze.cols
        v.cam.lookat[:] = (cols * CELL / 2, -rows * CELL / 2, 0.0)
        v.cam.distance = max(rows, cols) * CELL * 1.7
        v.cam.elevation, v.cam.azimuth = -70.0, 90.0
        app.new_episode()
        played, done_at = 1, None
        while v.is_running():
            app.handle_keys()
            if app.policy is not None:
                if app.env.done:
                    done_at = done_at or time.perf_counter()
                    if played < args.episodes and time.perf_counter() - done_at > 1.5:
                        app.seed += 1
                        app.new_episode()
                        played, done_at = played + 1, None
                else:
                    app.policy_step()
            app.draw()
            time.sleep(0.01)
            if args.exit_after is not None and time.perf_counter() - t_open > args.exit_after:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
