import copy
from collections.abc import Mapping
from pathlib import Path

from src.config.files import load_yaml
from src.config.train import normalize_train_config
from src.train.checkpoint import resolve_checkpoint
from src.train.relocate import relocate_checkpoint
from src.train.state import load_training


def check_resume_args(
    resume: Path | None, path_map: list | None, config: Path | Mapping | None
) -> None:
    if path_map and resume is None:
        raise ValueError("path_map requires --resume.")
    if resume is not None and config is not None:
        raise ValueError(
            "--resume uses saved settings; only --steps may override them."
        )
    if resume is None and config is None:
        raise ValueError("a new run needs a training config.")


def load_source(config: Path | Mapping, stage: str) -> dict:
    """Return the training YAML that a new run saves as its train.yaml."""
    if isinstance(config, Mapping):
        source = copy.deepcopy(dict(config))
    else:
        source = load_yaml(config)
    # A config may leave its stage implicit, but resume reads it from train.yaml.
    source.setdefault("stage", stage)
    return source


def load_resume(
    resume: Path, stage: str, path_map: list | None
) -> tuple[dict, dict, dict]:
    """Return the relocated payload, its resolved config, and the train.yaml to save."""
    checkpoint = resolve_checkpoint(resume)
    payload = relocate_checkpoint(load_training(checkpoint, stage), path_map)
    cfg = normalize_train_config(payload["config"], stage)
    # Save the resolved checkpoint config: it carries any relocated paths.
    return payload, cfg, copy.deepcopy(payload["config"])


def set_total_steps(
    cfg: dict, source: dict, steps: int | None, payload: dict | None
) -> None:
    """Apply a --steps override; a resumed run must have steps left to train."""
    if steps is not None:
        cfg["train"]["total_steps"] = steps
        source.setdefault("train", {})["total_steps"] = steps
    if payload is not None and cfg["train"]["total_steps"] <= payload["step"]:
        raise ValueError("total steps must exceed completed steps.")
