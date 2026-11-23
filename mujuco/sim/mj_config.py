"""Load configs/sim_default.yaml (robot choice, controller gains, noise) with overrides."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable

import yaml

from .noise import NoiseConfig

MJ_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_MJ_CONFIG = MJ_CONFIG_DIR / "sim_default.yaml"
CONTROLLER_KEYS = ("kp", "v_max", "eps_v", "eps_x", "t_min", "timeout")
ROBOTS = ("mecanum", "point")


def _merge(base: dict, extra: dict, where: str = "") -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        if key not in base:
            raise ValueError(f"unknown MuJoCo config key {where}{key!r}")
        out[key] = _merge(base[key], value, f"{where}{key}.") if isinstance(base[key], dict) else value
    return out


def load_mj_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Defaults, then ``path`` merged on top, then ``a.b.c=value`` overrides
    (e.g. ``robot=point``, ``noise.friction_on=true``, ``mecanum.noise.slip_on=true``).

    Returns a plain dict (picklable, YAML-friendly).
    """
    data = yaml.safe_load(DEFAULT_MJ_CONFIG.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_MJ_CONFIG.resolve():
        data = _merge(data, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        nested: Any = yaml.safe_load(raw)
        for part in reversed(key.strip().split(".")):
            nested = {part: nested}
        data = _merge(data, nested)
    check_mj_config(data)
    return data


def check_mj_config(data: dict[str, Any]) -> None:
    from .mecanum import mecanum_config_fields

    if set(data) != {"robot", "mecanum", "controller", "noise"}:
        raise ValueError(f"MuJoCo config sections must be robot, mecanum, controller, noise; got {sorted(data)}")
    if data["robot"] not in ROBOTS:
        raise ValueError(f"robot must be one of {ROBOTS}, got {data['robot']!r}")
    if set(data["controller"]) != set(CONTROLLER_KEYS):
        raise ValueError(f"controller keys must be {CONTROLLER_KEYS}, got {sorted(data['controller'])}")
    noise_fields = {f.name for f in dataclasses.fields(NoiseConfig)}
    if set(data["noise"]) != noise_fields:
        raise ValueError(f"noise keys must be {sorted(noise_fields)}, got {sorted(data['noise'])}")
    for section, fields in mecanum_config_fields().items():
        got = set(data["mecanum"].get(section) or {})
        if got != fields:
            raise ValueError(f"mecanum.{section} keys must be {sorted(fields)}, got {sorted(got)}")


def noise_config(data: dict[str, Any]) -> NoiseConfig:
    return NoiseConfig(**data["noise"])
