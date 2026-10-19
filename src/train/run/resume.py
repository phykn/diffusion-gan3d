import copy
from pathlib import Path

from src.config.train import normalize_train_config
from src.train.checkpoint import resolve_checkpoint
from src.train.relocate import relocate_checkpoint
from src.train.state import load_training


def check_resume_args(
    resume: Path | None, path_map: list | None, config: Path | None, data: Path | None
) -> None:
    if path_map and resume is None:
        raise ValueError("path_map requires --resume.")
    if resume is not None and (config is not None or data is not None):
        raise ValueError(
            "--resume uses saved settings; only --steps may override them."
        )


def load_resume(
    resume: Path, stage: str, path_map: list | None
) -> tuple[dict, dict, dict]:
    """Return the relocated payload, its resolved config, and the train.yaml to save."""
    checkpoint = resolve_checkpoint(resume)
    payload = relocate_checkpoint(load_training(checkpoint, stage), path_map)
    cfg = normalize_train_config(payload["config"], stage)
    # Save the resolved checkpoint config: it carries any relocated paths.
    return payload, cfg, copy.deepcopy(payload["config"])


def set_total_steps(cfg: dict, source: dict, steps: int | None) -> None:
    if steps is not None:
        cfg["train"]["total_steps"] = steps
        source.setdefault("train", {})["total_steps"] = steps
