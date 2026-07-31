import os
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

WEIGHTS_NAME = "model.pt"
CRITIC_NAMES = tuple(f"critic_{axis}.pt" for axis in range(3))


def save_weights(
    run_dir: str | Path,
    model: nn.Module,
) -> Path:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module.")

    root = Path(run_dir)
    path = root / WEIGHTS_NAME
    _save_atomic(model.state_dict(), path)
    return path


def save_training_weights(
    run_dir: str | Path,
    model: nn.Module,
    critics: nn.ModuleDict,
) -> Path:
    _validate_critics(critics)
    root = Path(run_dir)
    for axis, name in enumerate(CRITIC_NAMES):
        _save_atomic(critics[str(axis)].state_dict(), root / name)
    return save_weights(root, model)


def load_weights(
    weights: str | Path,
    model: nn.Module,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module.")
    state = _load_state(Path(weights))
    model.load_state_dict(state, strict=True)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"weights file does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"weights file must contain a state dict: {path}")
    return state


def _validate_critics(critics: nn.ModuleDict) -> None:
    if not isinstance(critics, nn.ModuleDict):
        raise TypeError("critics must be a ModuleDict.")
    if set(critics) != {"0", "1", "2"}:
        raise ValueError("critics must contain axes 0, 1, and 2.")


def _save_atomic(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
