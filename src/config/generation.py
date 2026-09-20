import math
from dataclasses import dataclass

from src.config.files import PROJECT_ROOT, load_yaml


@dataclass(frozen=True)
class GenerationSettings:
    guidance: float = 1.0
    anchor_strength: float = 1.0
    overlap: int = 8


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(PROJECT_ROOT / "config" / "gen.yaml")
    unknown = set(section) - {
        "guidance",
        "anchor_strength",
        "overlap",
    }
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown generation setting: {names}")

    defaults = GenerationSettings()
    guidance = section.get("guidance", defaults.guidance)
    if type(guidance) not in (int, float) or not math.isfinite(guidance):
        raise ValueError("guidance must be a finite number.")
    anchor_strength = section.get("anchor_strength", defaults.anchor_strength)
    if (
        not isinstance(anchor_strength, (int, float))
        or isinstance(anchor_strength, bool)
        or not math.isfinite(anchor_strength)
        or not 0.0 <= anchor_strength <= 1.0
    ):
        raise ValueError("anchor_strength must be between zero and one.")
    overlap = section.get("overlap", defaults.overlap)
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    return GenerationSettings(
        guidance=float(guidance),
        anchor_strength=float(anchor_strength),
        overlap=overlap,
    )
