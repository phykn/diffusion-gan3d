import os
import shutil
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

GENERATOR_FILE = "generator.pt"
CRITIC_FILES = tuple(f"critic_{axis}.pt" for axis in range(3))
CRITIC_C_FILE = "critic_c.pt"
CHECKPOINT_DIR = "checkpoints"


def save_weights(
    run_dir: str | Path,
    model: nn.Module,
) -> Path:
    root = Path(run_dir)
    path = root / GENERATOR_FILE
    save_atomic(model.state_dict(), path)
    return path


def save_all_weights(
    run_dir: str | Path,
    model: nn.Module,
    critics: nn.ModuleDict,
    connectivity_critic: nn.Module,
) -> Path:
    check_critics(critics)
    root = Path(run_dir)
    for axis in sorted(int(value) for value in critics):
        save_atomic(critics[str(axis)].state_dict(), root / CRITIC_FILES[axis])
    save_atomic(
        connectivity_critic.state_dict(),
        root / CRITIC_C_FILE,
    )
    return save_weights(root, model)


def load_all_weights(
    run_dir: str | Path,
    model: nn.Module,
    average: nn.Module,
    critics: nn.ModuleDict,
    connectivity_critic: nn.Module,
) -> None:
    check_critics(critics)
    root = Path(run_dir)
    generator = load_state(root / GENERATOR_FILE)
    model.load_state_dict(generator, strict=True)
    average.load_state_dict(generator, strict=True)
    for axis in sorted(int(value) for value in critics):
        load_weights(root / CRITIC_FILES[axis], critics[str(axis)])
    load_weights(root / CRITIC_C_FILE, connectivity_critic)


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: nn.Module,
    critics: nn.ModuleDict,
    connectivity_critic: nn.Module,
) -> Path:
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("checkpoint step must be a positive integer.")
    root = Path(run_dir)
    parent = root / CHECKPOINT_DIR
    target = parent / f"step_{step:08d}"
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")

    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{target.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        save_all_weights(
            temporary,
            model,
            critics,
            connectivity_critic,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target / GENERATOR_FILE


def load_weights(
    path: str | Path,
    model: nn.Module,
) -> None:
    state = load_state(Path(path))
    model.load_state_dict(state, strict=True)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"weights file does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"weights file must contain a state dict: {path}")
    return state


def check_critics(critics: nn.ModuleDict) -> None:
    if not isinstance(critics, nn.ModuleDict):
        raise TypeError("critics must be a ModuleDict.")
    if not critics or not set(critics).issubset({"0", "1", "2"}):
        raise ValueError("critics must contain one or more valid axes.")


def save_atomic(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(data, tmp)
        # Replacement exposes either the old complete checkpoint or the new one.
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
