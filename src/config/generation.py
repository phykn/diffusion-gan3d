import math
from dataclasses import dataclass, fields

from src.config.files import PROJECT_ROOT, load_yaml
from src.config.values import check_integer, check_number


@dataclass(frozen=True)
class GenerationSettings:
    guidance: float = 1.0
    anchor_strength: float = 1.0
    overlap: int = 8


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(PROJECT_ROOT / "config" / "gen.yaml")
    unknown = set(section) - {field.name for field in fields(GenerationSettings)}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown generation setting: {names}")

    defaults = GenerationSettings()
    guidance = section.get("guidance", defaults.guidance)
    if type(guidance) not in (int, float) or not math.isfinite(guidance):
        raise ValueError("guidance must be a finite number.")
    anchor_strength = section.get("anchor_strength", defaults.anchor_strength)
    check_number(anchor_strength, "anchor_strength")
    overlap = section.get("overlap", defaults.overlap)
    check_integer(overlap, "overlap", 0)
    return GenerationSettings(
        guidance=float(guidance),
        anchor_strength=float(anchor_strength),
        overlap=overlap,
    )
