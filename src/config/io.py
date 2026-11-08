import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.config.schema import StageConfig, build_config, to_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    """Resolve a configured path; relative paths follow the project root."""
    return (PROJECT_ROOT / Path(path).expanduser()).resolve()


def stored_path(path: str | Path) -> str:
    """Path as saved in settings: project-relative inside the root, else absolute."""
    resolved = project_path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


class _Dumper(yaml.SafeDumper):
    def represent_list(self, values):
        return self.represent_sequence("tag:yaml.org,2002:seq", values, flow_style=True)


_Dumper.add_representer(list, _Dumper.represent_list)


def dump_yaml(data: Mapping | StageConfig) -> str:
    """One YAML block per top-level key; short lists stay on one line."""
    values = to_dict(data)
    blocks = [
        yaml.dump({key: value}, Dumper=_Dumper, sort_keys=False)
        for key, value in values.items()
    ]
    return "\n".join(blocks) if blocks else "{}\n"


def save_yaml(path: str | Path, data: Mapping | StageConfig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data), encoding="utf-8")
    return path


def load_config(path: str | Path) -> StageConfig:
    return build_config(load_yaml(path))


@dataclass(frozen=True)
class GenerationSettings:
    """Defaults for generation; values come only from config/gen.yaml."""

    guidance: float
    anchor_strength: float
    overlap: int


def load_generation_settings() -> GenerationSettings:
    path = PROJECT_ROOT / "config" / "gen.yaml"
    section = load_yaml(path)
    names = {item.name for item in fields(GenerationSettings)}
    unknown = set(section) - names
    if unknown:
        raise ValueError(f"unknown generation setting: {', '.join(sorted(unknown))}")
    missing = names - set(section)
    if missing:
        raise ValueError(f"{path} must set {', '.join(sorted(missing))}.")
    guidance = section["guidance"]
    if type(guidance) not in (int, float) or not math.isfinite(guidance):
        raise ValueError("guidance must be a finite number.")
    strength = section["anchor_strength"]
    if (
        type(strength) not in (int, float)
        or not math.isfinite(strength)
        or strength < 0
    ):
        raise ValueError("anchor_strength must be finite and non-negative.")
    overlap = section["overlap"]
    if type(overlap) is not int or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    return GenerationSettings(float(guidance), float(strength), overlap)
