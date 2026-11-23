"""viewer.py logic with a fake window handle (no GUI): keys drive the env, policies play back,
the overlay and the texts are built."""

import contextlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewer import KEY_BELIEF, KEY_NEW, ViewerApp  # noqa: E402
from sim.mj_config import load_mj_config  # noqa: E402
from motion_model import Outcome  # noqa: E402
from runner import resolve_maze  # noqa: E402
from sim_config import load_config  # noqa: E402

UP, RIGHT, ENTER = 265, 262, 257


class FakeHandle:
    def __init__(self, model):
        self.user_scn = mujoco.MjvScene(model, maxgeom=2000)
        self.texts, self.n_sync = None, 0

    def lock(self):
        return contextlib.nullcontext()

    def set_texts(self, texts):
        self.texts = texts

    def sync(self):
        self.n_sync += 1


def _app(policy=None, robot="mecanum", speed=1e6, **kw):
    cfg = load_config(overrides=["maze.name=small/m5_a"])
    app = ViewerApp(cfg, resolve_maze(cfg), load_mj_config(overrides=[f"robot={robot}"]), policy, speed=speed, **kw)
    app.handle = FakeHandle(app.sim.model)
    app.sim.on_step = app.on_step
    app.new_episode()
    return app


@pytest.mark.parametrize("robot", ["mecanum", "point"])
def test_manual_keys_drive_the_env_and_update_the_belief(robot):
    app = _app(robot=robot, start=(4, 0))                      # bottom-left corner of m5_a
    for key in (UP, RIGHT, UP):
        app.key_callback(key)
    app.handle_keys()
    assert app.env.log.steps >= 1
    assert np.isclose(app.belief().sum(), 1.0)
    app.draw()
    assert app.handle.n_sync > 0 and app.handle.user_scn.ngeom >= 2
    assert "observation" in app.handle.texts[0][2]


def test_declare_and_new_episode_keys():
    app = _app(start=(0, 0))
    app.key_callback(ENTER)
    app.handle_keys()
    assert app.env.log.outcome is Outcome.WRONG_DECLARATION
    app.key_callback(KEY_NEW)
    app.handle_keys()
    assert app.seed == 1 and not app.env.done
    app.key_callback(KEY_BELIEF)
    app.handle_keys()
    app.overlay(app.handle.user_scn)
    assert app.handle.user_scn.ngeom == len(app.maze.goal_cells)   # belief hidden: goals only


def test_policy_playback_runs_an_episode_to_the_end():
    app = _app(policy="projecta", seed=2, speed=8.0)
    for _ in range(200):
        if app.env.done:
            break
        app.policy_step()
    assert app.env.done and app.env.log.outcome is Outcome.SUCCESS
    assert app.belief() is app.policy.agent.belief or np.allclose(app.belief(), app.policy.agent.belief)
    assert app.handle.n_sync > 10                                   # frames were shown during the moves
