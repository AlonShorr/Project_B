"""Tests for t5: one config file, reproducible headless runs, CSV output."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import config
import policies
import runner
import sim_config
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from motion_model import Action, MotionParams, Outcome, set_wall
from runner import LoadedMaze
from sim_config import ConfigError, SimConfig, load_config

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dfs_maze(rows: int, cols: int, seed: int, loops: int = 0) -> np.ndarray:
    """Recursive-backtracker maze (all cells connected), plus ``loops`` extra openings."""
    rng = np.random.default_rng(seed)
    wall_map = np.zeros((rows, cols, 4), dtype=np.uint8)
    steps = ((-1, 0, WALL_UP), (1, 0, WALL_DOWN), (0, -1, WALL_LEFT), (0, 1, WALL_RIGHT))
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc, w) for dr, dc, w in steps
                if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in seen]
        if not nbrs:
            stack.pop()
            continue
        nr, nc, w = nbrs[int(rng.integers(len(nbrs)))]
        set_wall(wall_map, r, c, w, closed=False)
        seen.add((nr, nc))
        stack.append((nr, nc))
    for _ in range(loops):
        r, c = int(rng.integers(rows - 1)), int(rng.integers(cols - 1))
        set_wall(wall_map, r, c, WALL_RIGHT if rng.random() < 0.5 else WALL_DOWN, closed=False)
    return wall_map


MAZE10 = LoadedMaze("test/dfs10", dfs_maze(10, 10, seed=3, loops=10), ((4, 4), (4, 5), (5, 4), (5, 5)))


def small_cfg(*overrides: str, out: Path | None = None) -> SimConfig:
    items = ["run.n_runs=60", "run.seed0=100", *overrides]
    if out is not None:
        items.append(f"run.out={out.as_posix()}")
    return load_config(overrides=items)


def read_rows(path: Path, drop=("wall_time_ms",)) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in drop:
            row.pop(k, None)
    return rows


def csv_text_without(path: Path, column: str = "wall_time_ms") -> str:
    """The CSV file with one column removed, as text (for byte comparisons)."""
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    i = header.index(column)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for line in [header, *reader]:
        w.writerow(line[:i] + line[i + 1:])
    return out.getvalue()


# ---------------------------------------------------------------------------
# t5.0: one config file
# ---------------------------------------------------------------------------

def test_default_yaml_matches_config_py():
    cfg = load_config()
    expected = MotionParams.from_config()
    got = cfg.to_motion_params()
    assert np.allclose(got.p_steps, expected.p_steps, atol=1e-12)
    assert got.p_drift == expected.p_drift and got.wall_mode == expected.wall_mode
    assert cfg.motion.cell_size_mm == config.CELL_SIZE_MM
    assert cfg.motion.step_sigma_mm == config.FORWARD_SIGMA_MM
    assert cfg.motion.step_bias_mm == config.FORWARD_BIAS_MM
    assert cfg.motion.extra_stay == config.FORWARD_EXTRA_STAY_PROB
    assert cfg.motion.extra_over == config.FORWARD_EXTRA_OVERSHOOT_PROB
    assert cfg.episode.max_steps == config.MAX_EPISODE_STEPS
    assert cfg.projecta.tau_loc == config.LOCALIZATION_THRESHOLD
    assert cfg.projecta.rho_amb == config.LOCALIZATION_AMBIGUITY_RATIO
    assert cfg.projecta.tau_lost == config.IS_LOST_THRESHOLD
    assert cfg.maze.name == "medium/m10_a"


def test_default_yaml_lists_every_parameter():
    """The YAML file spells out every field, and equals the dataclass defaults."""
    raw = sim_config.read_yaml(sim_config.DEFAULT_CONFIG_PATH)
    for name, cls in sim_config.SECTIONS.items():
        fields = {f.name for f in sim_config.dataclasses.fields(cls)}
        assert set(raw[name]) == fields, name
    for pname, pcls in policies.POLICIES.items():
        assert set(raw["policy"]["params"][pname]) == set(pcls.DEFAULTS), pname


def test_missing_keys_rejected():
    """The dataclasses carry no defaults: a complete dict is required."""
    full = load_config().to_dict()
    for section, key in (("motion", "p_drift"), ("episode", "max_steps"), ("sensing", "R")):
        data = load_config().to_dict()
        del data[section][key]
        with pytest.raises(ConfigError, match=key):
            SimConfig.from_dict(data)
    data = dict(full)
    del data["projecta"]
    with pytest.raises(ConfigError):
        SimConfig.from_dict(data)
    data = load_config().to_dict()
    del data["policy"]["params"]["random"]["declare_after"]
    with pytest.raises(ConfigError, match="declare_after"):
        SimConfig.from_dict(data)
    data = load_config().to_dict()
    del data["policy"]["params"]["oracle_bfs"]
    with pytest.raises(ConfigError, match="oracle_bfs"):
        SimConfig.from_dict(data)


def test_default_yaml_is_the_only_source_of_defaults(tmp_path):
    """Changing a default needs only the YAML; null sections keep the YAML values."""
    raw = sim_config.read_yaml(sim_config.DEFAULT_CONFIG_PATH)
    raw["motion"]["p_drift"] = 0.3
    raw["motion"]["cell_size_mm"] = 200.0
    raw["episode"]["max_steps"] = 77
    raw["policy"]["params"]["random"]["declare_after"] = 9
    base = tmp_path / "base.yaml"
    base.write_text(sim_config.yaml.safe_dump(raw), encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text("motion:\nepisode: null\npolicy:\n  params:\n    random:\n", encoding="utf-8")

    cfg = load_config(exp, base_path=base)
    assert cfg.motion.p_drift == 0.3 and cfg.motion.cell_size_mm == 200.0
    assert cfg.episode.max_steps == 77
    assert cfg.policy.params["random"]["declare_after"] == 9
    assert cfg.to_motion_params().p_drift == 0.3


def test_cell_size_is_a_config_parameter():
    base = load_config()
    cfg = load_config(overrides=["motion.cell_size_mm=120"])
    assert cfg.motion.resolved_p_steps() != base.motion.resolved_p_steps()
    assert cfg.config_hash() != base.config_hash()
    # Scaling cell size, sigma and bias together leaves p_steps unchanged.
    scaled = load_config(overrides=["motion.cell_size_mm=360", "motion.step_sigma_mm=30"])
    assert np.allclose(scaled.motion.resolved_p_steps(), base.motion.resolved_p_steps())


@pytest.mark.parametrize(
    "data",
    [
        {"motion": {"p_drfit": 0.2}},
        {"motoin": {"p_drift": 0.2}},
        {"policy": {"params": {"random": {"declare_aftr": 3}}}},
        {"policy": {"params": {"randm": {}}}},
        {"policy": {"name": "nope"}},
    ],
)
def test_unknown_key_rejected(tmp_path, data):
    path = tmp_path / "bad.yaml"
    path.write_text(sim_config.yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_key_message_suggests_fix():
    with pytest.raises(ConfigError, match="p_drift"):
        load_config(overrides=["motion.p_drfit=0.2"])


@pytest.mark.parametrize(
    "override",
    ["motion.p_drift=1.5", "motion.p_steps=[0.5, 0.6, 0.1]", "motion.wall_mode=bounce",
     "sensing.rays=5", "episode.max_steps=0", "run.workers=0", "motion.p_drift=abc",
     "motion.extra_stay=0.95", "motion.cell_size_mm=0", "episode.max_steps=2.5",
     "policy.params.random.declare_after=2.5", "policy.params.random.declare_after=true",
     "policy.params.random.diagonal=1", "motion=null"],
)
def test_invalid_values_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_override_and_partial_file(tmp_path):
    base = load_config()
    cfg = load_config(overrides=["motion.p_drift=0.2", "motion.p_steps=[0.1, 0.8, 0.1]", "maze.goal=[[1, 2]]"])
    assert cfg.motion.p_drift == 0.2
    assert cfg.to_motion_params() == MotionParams((0.1, 0.8, 0.1), 0.2, "collide")
    assert cfg.maze.goal == ((1, 2),)
    assert cfg.config_hash() != base.config_hash()

    path = tmp_path / "exp.yaml"
    path.write_text("motion: {p_drift: 0.2, p_steps: [0.1, 0.8, 0.1]}\nmaze: {goal: [[1, 2]]}\n", encoding="utf-8")
    from_file = load_config(path)
    assert from_file == cfg and from_file.config_hash() == cfg.config_hash()
    assert from_file.sensing == base.sensing   # everything else comes from default.yaml


def test_integer_values_checked_the_same_everywhere():
    cfg = load_config(overrides=["episode.max_steps=400.0", "policy.params.random.declare_after=20.0"])
    assert cfg.episode.max_steps == 400 and isinstance(cfg.episode.max_steps, int)
    assert cfg.policy.params["random"]["declare_after"] == 20
    assert isinstance(cfg.policy.params["random"]["declare_after"], int)
    assert load_config().config_hash() == cfg.config_hash()


def test_config_hash_ignores_run_section_and_unused_policy_params():
    base = load_config()
    same = load_config(overrides=["run.n_runs=5", "run.seed0=7", "run.out=x.csv", "run.workers=3",
                                  "policy.params.random.declare_after=3"])
    assert same.config_hash() == base.config_hash()
    assert load_config(overrides=["policy.params.oracle_bfs.diagonal=false"]).config_hash() != base.config_hash()
    assert load_config(overrides=["policy.name=random"]).config_hash() != base.config_hash()
    # int vs float spelling does not change the hash
    assert load_config(overrides=["motion.p_drift=0"]).config_hash() == load_config(
        overrides=["motion.p_drift=0.0"]).config_hash()


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def test_oracle_is_perfect_without_noise():
    cfg = small_cfg("motion.p_steps=[0, 1, 0]", "motion.p_drift=0")
    dist = policies.bfs_distances(MAZE10.wall_map, MAZE10.goal_cells)
    rows = runner.run_batch(cfg, MAZE10)
    assert all(r["outcome"] == "success" for r in rows)
    assert all(r["T"] == dist[r["start_r"], r["start_c"]] for r in rows)


def test_oracle_cache_follows_maze_content():
    policy = policies.make_policy("oracle_bfs")
    other = dfs_maze(10, 10, seed=11)
    for wall_map in (MAZE10.wall_map, other, MAZE10.wall_map.copy(), other.copy()):
        env = runner.GridEnv(wall_map, MAZE10.goal_cells, params=MotionParams.deterministic())
        policy.reset(policies.EnvInfo.from_env(env), np.random.default_rng(0))
        assert np.array_equal(policy._dist, policies.bfs_distances(wall_map, MAZE10.goal_cells))


def test_random_policy_declares_after_n_moves():
    cfg = small_cfg("policy.name=random", "policy.params.random.declare_after=7", "motion.wall_mode=stop")
    rows = runner.run_batch(cfg, MAZE10)
    assert all(r["outcome"] in ("success", "wrong_declaration") for r in rows)
    assert all(r["n_moves"] == 7 and r["T"] == 7.0 for r in rows)


def test_non_privileged_policy_does_not_see_true_cell():
    seen = []

    class Spy(policies.Policy):
        name = "random"
        DEFAULTS = {"declare_after": 3, "diagonal": True}

        def act(self, obs, info):
            seen.append(dict(info))
            return Action.DECLARE if len(seen) > 2 else Action.N

    cfg = small_cfg("motion.wall_mode=stop", "run.n_runs=1")
    er = runner.EpisodeRunner(cfg, MAZE10)
    er.policy = Spy()
    er.run(0)
    assert seen and all("cell" not in info for info in seen)


# ---------------------------------------------------------------------------
# t5.1: reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_csv(tmp_path):
    for policy in ("random", "oracle_bfs"):
        a = runner.run_and_write(small_cfg(f"policy.name={policy}", out=tmp_path / f"a_{policy}.csv"), MAZE10)[0]
        b = runner.run_and_write(small_cfg(f"policy.name={policy}", out=tmp_path / f"b_{policy}.csv"), MAZE10)[0]
        assert csv_text_without(a) == csv_text_without(b)


def test_worker_count_does_not_change_output(tmp_path):
    cfg1 = small_cfg("policy.name=random", "run.n_runs=200", out=tmp_path / "w1.csv")
    cfg2 = small_cfg("policy.name=random", "run.n_runs=200", "run.workers=2", out=tmp_path / "w2.csv")
    a = runner.run_and_write(cfg1, MAZE10)[0]
    b = runner.run_and_write(cfg2, MAZE10)[0]
    assert csv_text_without(a) == csv_text_without(b)
    assert [r["seed"] for r in read_rows(b)] == [str(s) for s in range(100, 300)]


def test_policy_change_keeps_start_cells():
    rand = runner.run_batch(small_cfg("policy.name=random"), MAZE10)
    oracle = runner.run_batch(small_cfg("policy.name=oracle_bfs"), MAZE10)
    starts = lambda rows: [(r["seed"], r["start_r"], r["start_c"]) for r in rows]
    assert starts(rand) == starts(oracle)
    assert len(set(starts(rand))) > 1


def test_policy_rng_is_independent_of_env_rng():
    seed = 5
    env_draws = np.random.default_rng(seed).random(5)
    pol_draws = runner.policy_rng(seed).random(5)
    assert not np.allclose(env_draws, pol_draws)
    assert np.array_equal(runner.policy_rng(seed).random(5), pol_draws)


def test_different_seeds_differ():
    rows = runner.run_batch(small_cfg("policy.name=random", "run.n_runs=50"), MAZE10)
    assert len({(r["start_r"], r["start_c"], r["final_r"], r["final_c"]) for r in rows}) > 10


# ---------------------------------------------------------------------------
# t5.3: CSV output
# ---------------------------------------------------------------------------

def test_csv_columns_and_sidecar(tmp_path):
    out = tmp_path / "res" / "run.csv"
    cfg = small_cfg("run.n_runs=10", "motion.p_drift=0.2", out=out)
    runner.run_and_write(cfg, MAZE10)
    rows = read_rows(out, drop=())
    assert len(rows) == 10
    header = list(rows[0])
    for col in ("run_id", "seed", "maze", "policy", "policy_params_json", "T", "outcome", "steps_to_loc",
                "final_P_goal", "n_fallbacks", "wall_time_ms", "config_hash", "motion.p_drift",
                "motion.p_step0", "motion.wall_mode", "sensing.R", "sensing.p_miss", "episode.max_steps",
                "start_r", "start_c", "final_r", "final_c"):
        assert col in header, col
    assert header == runner.csv_columns(cfg)
    row = rows[0]
    assert row["config_hash"] == runner.experiment_hash(cfg, MAZE10) and row["motion.p_drift"] == "0.2"
    assert row["maze"] == MAZE10.name and row["maze_hash"] == MAZE10.maze_hash
    assert row["run_id"] == f"{row['config_hash']}-{row['seed']}"
    assert row["steps_to_loc"] == "" and row["n_fallbacks"] == ""
    assert {r["outcome"] for r in rows} <= {o.value for o in Outcome}
    assert len({r["run_id"] for r in rows}) == 10

    side = runner.sidecar_path(out)
    assert side.name == "run.config.yaml" and side.exists()
    text = side.read_text(encoding="utf-8")
    assert f"# maze_name: {MAZE10.name}" in text and f"# maze_hash: {MAZE10.maze_hash}" in text
    assert "# maze_goal_cells: [[4, 4], [4, 5], [5, 4], [5, 5]]" in text
    assert f"# config_hash: {row['config_hash']}" in text
    reloaded = load_config(side)
    assert reloaded.maze.name == MAZE10.name   # the maze actually run, not the YAML default
    assert reloaded.config_hash(MAZE10.maze_hash) == row["config_hash"]
    assert runner.experiment_hash(reloaded, MAZE10) == row["config_hash"]


def test_config_hash_covers_maze_content():
    cfg = small_cfg()
    other = LoadedMaze(MAZE10.name, dfs_maze(10, 10, seed=11), MAZE10.goal_cells)
    moved_goal = LoadedMaze(MAZE10.name, MAZE10.wall_map, ((0, 0),))
    hashes = {runner.experiment_hash(cfg, m) for m in (MAZE10, other, moved_goal)}
    assert len(hashes) == 3
    rows = runner.run_batch(small_cfg("run.n_runs=2"), other)
    assert rows[0]["config_hash"] == runner.experiment_hash(cfg, other)


def test_cli_end_to_end(tmp_path, caplog):
    out = tmp_path / "cli.csv"
    code = runner.main(["--n-runs", "20", "--seed0", "3", "--out", str(out), "--policy", "random",
                        "--set", "motion.p_drift=0.05", "--log-level", "WARNING"], maze=MAZE10)
    assert code == 0
    rows = read_rows(out)
    assert [r["seed"] for r in rows] == [str(s) for s in range(3, 23)]
    assert {r["policy"] for r in rows} == {"random"} and rows[0]["motion.p_drift"] == "0.05"
    assert runner.main(["--set", "motion.typo=1"], maze=MAZE10) == 2


def test_resolve_maze_uses_maze_bank_contract(monkeypatch):
    """resolve_maze calls maze_bank.load_maze(name) and applies the goal override."""
    import types

    calls = []
    spec = types.SimpleNamespace(name="medium/m10_a", wall_map=MAZE10.wall_map, goal_cells=((5, 5),),
                                 rows=10, cols=10)
    fake = types.SimpleNamespace(load_maze=lambda name: calls.append(name) or spec)
    monkeypatch.setitem(sys.modules, "maze_bank", fake)

    maze = runner.resolve_maze(load_config())
    assert calls == ["medium/m10_a"] and maze.name == "medium/m10_a" and maze.goal_cells == ((5, 5),)
    maze = runner.resolve_maze(load_config(overrides=["maze.goal=[[0, 0], [0, 1]]"]))
    assert maze.goal_cells == ((0, 0), (0, 1))
    with pytest.raises(ValueError):
        runner.resolve_maze(load_config(overrides=["maze.goal=[[10, 0]]"]))


def test_runner_does_not_import_pygame():
    code = "import sys, runner, sim_config, policies, grid_env; sys.exit('pygame' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=HERE).returncode == 0


# ---------------------------------------------------------------------------
# t5.2: 1,000 episodes on a 10x10 maze in well under 10 minutes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["oracle_bfs", "random"])
def test_thousand_episodes_10x10_is_fast(policy):
    cfg = load_config(overrides=["run.n_runs=1000", f"policy.name={policy}"])
    t0 = time.perf_counter()
    rows = runner.run_batch(cfg, MAZE10, workers=1)
    elapsed = time.perf_counter() - t0
    assert len(rows) == 1000 and MAZE10.rows == MAZE10.cols == 10
    assert elapsed < 60.0, f"1000 episodes took {elapsed:.1f} s"   # budget is 600 s


# ---------------------------------------------------------------------------
# t1.2 / t1.3: chosen defaults (docs/decisions/t1_defaults.md)
# ---------------------------------------------------------------------------

def test_t1_defaults():
    cfg = load_config()
    assert cfg.to_motion_params().p_steps == pytest.approx((0.05, 0.9499, 0.0001), abs=1e-6)
    assert cfg.motion.p_drift == pytest.approx(1e-4)
    assert cfg.sensing.R == 2 and cfg.sensing.rays == 4 and cfg.sensing.p_miss == pytest.approx(0.05)
    assert cfg.episode.max_steps == 400
    p = cfg.projecta
    assert (p.tau_loc, p.rho_amb, p.tau_lost, p.sep_min) == (0.25, 0.5, 0.15, 2)
    assert p.tau_declare == pytest.approx(0.95) and p.eps_col == pytest.approx(0.005)
    # the collision gate must let a well-localized robot move at all
    assert p.eps_col > cfg.motion.p_drift + cfg.to_motion_params().p_steps[2]
    assert cfg.evaluation.d_targets == (0.01, 0.05, 0.10)


def test_switch_needs_hysteresis():
    with pytest.raises(ConfigError, match="hysteresis"):
        load_config(overrides=["projecta.tau_lost=0.3"])


@pytest.mark.parametrize("bad", ["[]", "[0.05, 0.01]", "[0.0, 0.1]", "[0.01, 0.01]", "0.05"])
def test_bad_d_targets_rejected(bad):
    with pytest.raises(ConfigError, match="d_targets"):
        load_config(overrides=[f"evaluation.d_targets={bad}"])


def test_evaluation_section_not_in_hash():
    base = load_config()
    other = load_config(overrides=["evaluation.d_targets=[0.02, 0.2]"])
    assert base.config_hash("m") == other.config_hash("m")
    assert base.config_hash("m") != load_config(overrides=["projecta.sep_min=3"]).config_hash("m")


def test_stress_preset_loads():
    cfg = load_config(Path(sim_config.__file__).parent / "configs" / "stress.yaml")
    assert cfg.to_motion_params() == MotionParams((0.1, 0.898, 0.002), 0.002, "collide")
