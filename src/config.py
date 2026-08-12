import math
from dataclasses import dataclass
from pathlib import Path

from .utils import load_yaml


@dataclass(frozen=True)
class GenerationSettings:
    guidance_scale: float = 1.0
    overlap: int = 8
    crop_margin: int = 8


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(Path(__file__).resolve().parents[1] / "config" / "gen.yaml")
    unknown = section.keys() - {"guidance_scale", "overlap", "crop_margin"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown generation setting: {names}")

    guidance_scale = section.get("guidance_scale", 1.0)
    if (
        not isinstance(guidance_scale, (int, float))
        or isinstance(guidance_scale, bool)
        or not math.isfinite(guidance_scale)
        or guidance_scale < 0.0
        or guidance_scale > 10_000.0
    ):
        raise ValueError("generation.guidance_scale must be between zero and 10000.")
    overlap = _non_negative_int(section.get("overlap", 8), "overlap")
    crop_margin = _non_negative_int(section.get("crop_margin", 8), "crop_margin")
    return GenerationSettings(float(guidance_scale), overlap, crop_margin)


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"generation.{name} must be a non-negative integer.")
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
