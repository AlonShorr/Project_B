"""Experiment configuration for Project B (t5.0): YAML -> frozen dataclasses.

``configs/default.yaml`` is the only source of default values. The dataclasses
below have no defaults: every key must be present after merging. Another
config file only lists the keys it changes and is deep-merged on top of
``default.yaml`` (a null section or a null policy entry keeps the defaults),
and ``--set section.key=value`` overrides are applied last (values are parsed
as YAML, so ``0.2``, ``null``, ``[0.1, 0.8, 0.1]`` and ``[[4, 4]]`` all work).

Unknown keys raise ``ConfigError`` with the closest valid name, so a typo can
never silently fall back to a default. Missing keys raise as well.

``config_hash(maze_hash)`` identifies the experiment: a SHA-256 of the
canonical JSON of the resolved config plus the maze content hash. It leaves
out the ``run`` section (seeds, counts, output path and worker count do not
change the experiment), the ``evaluation`` section (the target failure rates d
only matter for the analysis) and the parameter entries of policies other than
the selected one.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from discrete_noise import gaussian_three_bin_probs, mix_extra_outcomes
from motion_model import WALL_MODES, MotionParams

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

DIAG_RULES = ("both_L_open",)
MISS_MODES = ("skip_to_next", "first_only")
RAY_COUNTS = (4, 8)
MAX_EXTRA_MASS = 0.95   # discrete_noise.mix_extra_outcomes caps extra_stay + extra_over here


class ConfigError(ValueError):
    """Invalid configuration: unknown or missing key, wrong type or out-of-range value."""


# --------------------------------------------------------------------------
# Value checks (shared with policies.py)
# --------------------------------------------------------------------------

def check_float(value: Any, name: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    if lo is not None and value < lo:
        raise ConfigError(f"{name} must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise ConfigError(f"{name} must be <= {hi}, got {value}")
    return value


def check_int(value: Any, name: str, lo: int | None = None) -> int:
    """An int, or a float with an integer value (``200.0``). Booleans are rejected."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if lo is not None and value < lo:
        raise ConfigError(f"{name} must be >= {lo}, got {value}")
    return int(value)


def check_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")
    return value


def check_choice(value: Any, name: str, choices: Iterable[Any]) -> Any:
    choices = tuple(choices)
    if value not in choices:
        raise ConfigError(f"{name} must be one of {choices}, got {value!r}")
    return value


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# --------------------------------------------------------------------------
# Sections (no defaults here: they come from configs/default.yaml)
# --------------------------------------------------------------------------

Cell = tuple[int, int]


@dataclass(frozen=True)
class MazeConfig:
    name: str
    goal: tuple[Cell, ...] | None   # None: use the maze's own goal cells

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError(f"maze.name must be a non-empty string, got {self.name!r}")
        if self.goal is not None:
            try:
                goal = tuple((int(r), int(c)) for r, c in self.goal)
            except (TypeError, ValueError):
                raise ConfigError(f"maze.goal must be a list of [r, c] pairs, got {self.goal!r}")
            if not goal:
                raise ConfigError("maze.goal must not be empty (use null for the maze default)")
            _set(self, "goal", goal)


@dataclass(frozen=True)
class MotionConfig:
    cell_size_mm: float
    step_sigma_mm: float
    step_bias_mm: float
    extra_stay: float
    extra_over: float
    p_steps: tuple[float, float, float] | None
    p_drift: float
    diag_rule: str
    wall_mode: str

    def __post_init__(self):
        _set(self, "cell_size_mm", check_float(self.cell_size_mm, "motion.cell_size_mm", lo=1e-9))
        _set(self, "step_sigma_mm", check_float(self.step_sigma_mm, "motion.step_sigma_mm", lo=0.0))
        _set(self, "step_bias_mm", check_float(self.step_bias_mm, "motion.step_bias_mm"))
        _set(self, "extra_stay", check_float(self.extra_stay, "motion.extra_stay", 0.0, 1.0))
        _set(self, "extra_over", check_float(self.extra_over, "motion.extra_over", 0.0, 1.0))
        if self.extra_stay + self.extra_over > MAX_EXTRA_MASS:
            raise ConfigError(
                f"motion.extra_stay + motion.extra_over must be <= {MAX_EXTRA_MASS}, "
                f"got {self.extra_stay + self.extra_over}"
            )
        _set(self, "p_drift", check_float(self.p_drift, "motion.p_drift", 0.0, 1.0))
        check_choice(self.diag_rule, "motion.diag_rule", DIAG_RULES)
        check_choice(self.wall_mode, "motion.wall_mode", WALL_MODES)
        if self.p_steps is not None:
            if isinstance(self.p_steps, (str, bytes)) or len(self.p_steps) != 3:
                raise ConfigError(f"motion.p_steps must be [p0, p1, p2] or null, got {self.p_steps!r}")
            p = tuple(check_float(x, "motion.p_steps", 0.0, 1.0) for x in self.p_steps)
            if not math.isclose(sum(p), 1.0, abs_tol=1e-9):
                raise ConfigError(f"motion.p_steps must sum to 1, got {p}")
            _set(self, "p_steps", p)

    def resolved_p_steps(self) -> tuple[float, float, float]:
        """p_steps as given, or from the Gaussian three-bin form (Project A)."""
        if self.p_steps is not None:
            return self.p_steps
        base = gaussian_three_bin_probs(
            self.cell_size_mm,
            self.cell_size_mm / 2.0,
            self.step_sigma_mm,
            self.step_bias_mm,
        )
        return mix_extra_outcomes(base, self.extra_stay, self.extra_over)


@dataclass(frozen=True)
class SensingConfig:
    R: int
    rays: int
    p_miss: float
    miss_mode: str

    def __post_init__(self):
        _set(self, "R", check_int(self.R, "sensing.R", lo=1))
        _set(self, "rays", check_choice(check_int(self.rays, "sensing.rays"), "sensing.rays", RAY_COUNTS))
        _set(self, "p_miss", check_float(self.p_miss, "sensing.p_miss", 0.0, 1.0))
        check_choice(self.miss_mode, "sensing.miss_mode", MISS_MODES)


@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int
    step_cost: float
    declare_cost: float

    def __post_init__(self):
        _set(self, "max_steps", check_int(self.max_steps, "episode.max_steps", lo=1))
        _set(self, "step_cost", check_float(self.step_cost, "episode.step_cost", lo=0.0))
        _set(self, "declare_cost", check_float(self.declare_cost, "episode.declare_cost", lo=0.0))


@dataclass(frozen=True)
class ProjectAConfig:
    tau_loc: float
    rho_amb: float
    tau_lost: float
    tau_declare: float
    eps_col: float
    move_cost: float
    sep_min: int

    def __post_init__(self):
        for name in ("tau_loc", "rho_amb", "tau_lost", "tau_declare", "eps_col"):
            _set(self, name, check_float(getattr(self, name), f"projecta.{name}", 0.0, 1.0))
        _set(self, "move_cost", check_float(self.move_cost, "projecta.move_cost", lo=0.0))
        _set(self, "sep_min", check_int(self.sep_min, "projecta.sep_min", lo=1))
        if self.tau_lost >= self.tau_loc:
            raise ConfigError(
                f"projecta.tau_lost ({self.tau_lost}) must be below projecta.tau_loc "
                f"({self.tau_loc}) so the localize/travel switch has hysteresis"
            )


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    # {policy name: {param: value}}. Only the entry for ``name`` is used, and
    # every entry must list all of that policy's parameters.
    params: Mapping[str, Mapping[str, Any]]

    def __post_init__(self):
        import policies  # local import: policies imports the checks from this module

        check_choice(self.name, "policy.name", policies.POLICY_NAMES)
        if not isinstance(self.params, Mapping):
            raise ConfigError(f"policy.params must be a mapping, got {self.params!r}")
        if self.name not in self.params:
            raise ConfigError(f"policy.params has no entry for the selected policy {self.name!r}")
        params = {}
        for pname, pparams in self.params.items():
            if pname not in policies.POLICY_NAMES:
                raise ConfigError(_unknown_msg("policy.params", pname, policies.POLICY_NAMES))
            try:
                params[pname] = policies.resolve_params(pname, pparams, strict=True)
            except ValueError as exc:
                raise ConfigError(f"policy.params.{pname}: {exc}") from None
        _set(self, "params", params)

    @property
    def selected_params(self) -> dict[str, Any]:
        """Parameters of the selected policy."""
        return dict(self.params[self.name])


@dataclass(frozen=True)
class RunConfig:
    n_runs: int
    seed0: int
    workers: int
    out: str

    def __post_init__(self):
        _set(self, "n_runs", check_int(self.n_runs, "run.n_runs", lo=1))
        _set(self, "seed0", check_int(self.seed0, "run.seed0", lo=0))
        _set(self, "workers", check_int(self.workers, "run.workers", lo=1))
        if not isinstance(self.out, str) or not self.out:
            raise ConfigError(f"run.out must be a non-empty path, got {self.out!r}")


@dataclass(frozen=True)
class EvaluationConfig:
    # Target failure rates d at which T is read off the Pareto frontier (t8).
    # They do not change a run, so they are not part of config_hash.
    d_targets: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.d_targets, (list, tuple)) or not self.d_targets:
            raise ConfigError(f"evaluation.d_targets must be a non-empty list, got {self.d_targets!r}")
        values = tuple(check_float(d, "evaluation.d_targets", 0.0, 1.0) for d in self.d_targets)
        if any(d <= 0.0 or d >= 1.0 for d in values) or list(values) != sorted(set(values)):
            raise ConfigError(
                f"evaluation.d_targets must be distinct, increasing and inside (0, 1), got {list(values)}"
            )
        _set(self, "d_targets", values)


SECTIONS: dict[str, type] = {
    "maze": MazeConfig,
    "motion": MotionConfig,
    "sensing": SensingConfig,
    "episode": EpisodeConfig,
    "projecta": ProjectAConfig,
    "policy": PolicyConfig,
    "run": RunConfig,
    "evaluation": EvaluationConfig,
}


@dataclass(frozen=True)
class SimConfig:
    maze: MazeConfig
    motion: MotionConfig
    sensing: SensingConfig
    episode: EpisodeConfig
    projecta: ProjectAConfig
    policy: PolicyConfig
    run: RunConfig
    evaluation: EvaluationConfig

    # -- construction --------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimConfig":
        """Build from a complete config dict (every section and key present)."""
        if not isinstance(data, Mapping):
            raise ConfigError(f"config must be a mapping, got {type(data).__name__}")
        _check_keys(data, SECTIONS, "config")
        sections = {}
        for name, section_cls in SECTIONS.items():
            raw = data.get(name)
            if not isinstance(raw, Mapping):
                raise ConfigError(f"section {name!r} must be a mapping, got {raw!r}")
            fields = [f.name for f in dataclasses.fields(section_cls)]
            _check_keys(raw, fields, name)
            missing = [f for f in fields if f not in raw]
            if missing:
                raise ConfigError(f"missing key(s) in {name}: {', '.join(missing)}")
            sections[name] = section_cls(**raw)
        return cls(**sections)

    def replace(self, overrides: Iterable[str] = ()) -> "SimConfig":
        """Return a copy with ``section.key=value`` overrides applied."""
        data = self.to_dict()
        apply_overrides(data, overrides)
        return SimConfig.from_dict(data)

    # -- views ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (lists instead of tuples), suitable for YAML and JSON."""
        return _plain(dataclasses.asdict(self))

    def hash_dict(self) -> dict[str, Any]:
        """The part of the config that defines the experiment (see module doc)."""
        data = self.to_dict()
        del data["run"]
        del data["evaluation"]
        data["policy"]["params"] = {self.policy.name: data["policy"]["params"][self.policy.name]}
        return data

    def config_hash(self, maze_hash: str | None = None) -> str:
        """Experiment id: the config (see ``hash_dict``) plus the maze content hash.

        ``maze_hash=None`` hashes the config alone; the runner always passes the
        hash of the maze it actually ran.
        """
        payload = {"config": self.hash_dict(), "maze_hash": maze_hash}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_motion_params(self) -> MotionParams:
        return MotionParams(
            p_steps=self.motion.resolved_p_steps(),
            p_drift=self.motion.p_drift,
            wall_mode=self.motion.wall_mode,
        )

    def flat_params(self) -> dict[str, Any]:
        """Experiment parameters as ``section.key`` columns for the CSV."""
        data = self.to_dict()
        flat: dict[str, Any] = {}
        for section in ("motion", "sensing", "episode", "projecta"):
            for key, value in data[section].items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, separators=(",", ":"))
                flat[f"{section}.{key}"] = value
            if section == "motion":   # the p_steps actually used, after the Gaussian form
                for k, p in enumerate(self.motion.resolved_p_steps()):
                    flat[f"motion.p_step{k}"] = p
        return flat

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None)


# --------------------------------------------------------------------------
# Loading and overrides
# --------------------------------------------------------------------------

def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _unknown_msg(where: str, key: Any, valid: Iterable[str]) -> str:
    valid = sorted(valid)
    close = difflib.get_close_matches(str(key), valid, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"unknown key {key!r} in {where}.{hint} Valid keys: {', '.join(valid)}"


def _check_keys(data: Mapping[str, Any], valid: Iterable[str], where: str) -> None:
    valid = list(valid)
    for key in data:
        if key not in valid:
            raise ConfigError(_unknown_msg(where, key, valid))


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` into a copy of ``base``.

    Mappings merge recursively. ``null`` over a mapping keeps the mapping (a
    bare ``motion:`` in a file means "no changes"). Everything else replaces.
    """
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(out.get(key), Mapping) and (value is None or isinstance(value, Mapping)):
            out[key] = deep_merge(dict(out[key]), value or {})
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value``; the value is parsed as YAML."""
    if "=" not in item:
        raise ConfigError(f"override must look like section.key=value, got {item!r}")
    path, raw = item.split("=", 1)
    keys = [k.strip() for k in path.strip().split(".")]
    if len(keys) < 2 or not all(keys):
        raise ConfigError(f"override key must look like section.key, got {path!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse value in override {item!r}: {exc}") from None
    return keys, value


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides to a raw config dict in place.

    Missing intermediate mappings are created; unknown keys are reported when
    the dict is turned into a ``SimConfig``.
    """
    for item in overrides:
        keys, value = parse_override(item)
        node = data
        for key in keys[:-1]:
            child = node.get(key)
            if child is None:
                child = node[key] = {}
            elif not isinstance(child, dict):
                raise ConfigError(f"override {item!r}: {key!r} is not a section")
            node = child
        if isinstance(node.get(keys[-1]), dict):
            raise ConfigError(f"override {item!r}: {keys[-1]!r} is a section; set its keys instead")
        node[keys[-1]] = value
    return data


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
    base_path: str | Path = DEFAULT_CONFIG_PATH,
) -> SimConfig:
    """Load ``base_path`` (default.yaml), merge ``path`` on top of it, then apply overrides."""
    data = read_yaml(base_path)
    if path is not None and Path(path).resolve() != Path(base_path).resolve():
        data = deep_merge(data, read_yaml(path))
    apply_overrides(data, overrides)
    return SimConfig.from_dict(data)
