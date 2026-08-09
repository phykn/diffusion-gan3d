"""Training configuration contracts shared by build and tooling code."""

from pathlib import Path

SCHEMA_VERSION = 2


def require_schema(cfg: dict) -> None:
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"train config schema_version must be {SCHEMA_VERSION}; "
            "old training configs are not supported."
        )


def get_schedule_steps(
    section: dict,
    name: str,
    total_steps: int,
) -> tuple[int, int]:
    start = section["start_step"]
    ramp = section["ramp_steps"]
    for field, value in (("start_step", start), ("ramp_steps", ramp)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name}.{field} must be a non-negative integer.")
    if start + ramp > total_steps:
        raise ValueError(f"{name} start and ramp steps must not exceed total steps.")
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
