import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import AXES
from .utils import load_yaml


def get_domains(
    data: Mapping[str, object],
) -> dict[int, dict[int, Sequence[str | Path]]]:
    domains = data["domains"]
    if not isinstance(domains, Mapping):
        raise TypeError("data.domains must be a mapping.")
    if not domains:
        raise ValueError("data.domains must not be empty.")
    if set(domains) != set(range(len(domains))):
        raise ValueError("domain IDs must be contiguous and start at zero.")
    parsed = {}
    for domain, folders in domains.items():
        if not isinstance(folders, Mapping):
            raise TypeError(f"domain {domain} must map axes to folders.")
        if not folders:
            raise ValueError(f"domain {domain} must contain at least one axis.")
        unknown = set(folders) - set(AXES)
        if unknown:
            raise ValueError(
                f"domain {domain} contains invalid axes: {sorted(unknown)}."
            )
        parsed[domain] = dict(folders)
    return parsed


@dataclass(frozen=True)
class GenerationSettings:
    guidance: float = 1.0
    anchor_strength: float = 1.0
    overlap: int = 8
    anchor_spread: float = 0.1


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(Path(__file__).resolve().parents[1] / "config" / "gen.yaml")
    unknown = set(section) - {
        "guidance",
        "anchor_strength",
        "anchor_spread",
        "overlap",
    }
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown generation setting: {names}")

    guidance = float(section.get("guidance", 1.0))
    anchor_strength = section.get("anchor_strength", 1.0)
    if (
        not isinstance(anchor_strength, (int, float))
        or isinstance(anchor_strength, bool)
        or not math.isfinite(anchor_strength)
        or not 0.0 <= anchor_strength <= 1.0
    ):
        raise ValueError("anchor_strength must be between zero and one.")
    anchor_spread = section.get("anchor_spread", 0.1)
    if (
        not isinstance(anchor_spread, (int, float))
        or isinstance(anchor_spread, bool)
        or not math.isfinite(anchor_spread)
        or anchor_spread <= 0.0
    ):
        raise ValueError("anchor_spread must be positive and finite.")
    overlap = section.get("overlap", 8)
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    return GenerationSettings(
        guidance=guidance,
        anchor_strength=float(anchor_strength),
        overlap=overlap,
        anchor_spread=float(anchor_spread),
    )


def get_schedule_steps(
    section: dict,
    name: str,
) -> tuple[int, int]:
    start = section["start_step"]
    ramp = section["ramp_steps"]
    for field, value in (("start_step", start), ("ramp_steps", ramp)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name}.{field} must be a non-negative integer.")
    return start, ramp


def find_train_config(weights: str | Path) -> Path:
    path = Path(weights).resolve()
    for parent in path.parents:
        config = parent / "train.yaml"
        if config.is_file():
            return config
    raise FileNotFoundError(f"train.yaml was not found above weights file: {path}")
