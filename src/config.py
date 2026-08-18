import math
from dataclasses import dataclass
from pathlib import Path

from .utils import load_yaml

DEFAULT_ANCHOR_STRENGTH = 0.90
DEFAULT_ANCHOR_SPREAD = 0.20


@dataclass(frozen=True)
class GenerationSettings:
    guidance: float = 1.0
    anchor_strength: float = DEFAULT_ANCHOR_STRENGTH
    overlap: int = 8
    anchor_spread: float = DEFAULT_ANCHOR_SPREAD


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(Path(__file__).resolve().parents[1] / "config" / "gen.yaml")
    unknown = section.keys() - {
        "guidance",
        "anchor_strength",
        "anchor_spread",
        "overlap",
    }
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown generation setting: {names}")

    guidance = section.get("guidance", 1.0)
    if (
        not isinstance(guidance, (int, float))
        or isinstance(guidance, bool)
        or not math.isfinite(guidance)
        or guidance < 0.0
        or guidance > 10_000.0
    ):
        raise ValueError("guidance must be between zero and 10000.")
    anchor_strength = section.get("anchor_strength", DEFAULT_ANCHOR_STRENGTH)
    if (
        not isinstance(anchor_strength, (int, float))
        or isinstance(anchor_strength, bool)
        or not math.isfinite(anchor_strength)
        or not 0.0 <= anchor_strength <= 1.0
    ):
        raise ValueError("anchor_strength must be between zero and one.")
    anchor_spread = section.get("anchor_spread", DEFAULT_ANCHOR_SPREAD)
    if (
        not isinstance(anchor_spread, (int, float))
        or isinstance(anchor_spread, bool)
        or not math.isfinite(anchor_spread)
        or anchor_spread <= 0.0
    ):
        raise ValueError("anchor_spread must be positive and finite.")
    overlap = _non_negative_int(section.get("overlap", 8), "overlap")
    return GenerationSettings(
        float(guidance),
        float(anchor_strength),
        overlap,
        float(anchor_spread),
    )


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


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
    """Find the run configuration that owns a generator weight file."""
    path = Path(weights).resolve()
    config = next(
        (
            parent / "train.yaml"
            for parent in path.parents
            if (parent / "train.yaml").is_file()
        ),
        None,
    )
    if config is None:
        raise FileNotFoundError(f"train.yaml was not found above weights file: {path}")
    return config.resolve()
