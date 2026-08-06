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


def find_weights(run_root: str | Path) -> Path:
    root = Path(run_root)
    paths = tuple(root.glob(f"*/{GENERATOR_FILE}"))
    if not paths:
        raise FileNotFoundError(f"no {GENERATOR_FILE} file was found under {root}.")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


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
    for axis, name in enumerate(CRITIC_FILES):
        save_atomic(critics[str(axis)].state_dict(), root / name)
    save_atomic(
        connectivity_critic.state_dict(),
        root / CRITIC_C_FILE,
    )
    return save_weights(root, model)


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
    if set(critics) != {"0", "1", "2"}:
        raise ValueError("critics must contain axes 0, 1, and 2.")


def save_atomic(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(data, tmp)
        # Replacement exposes either the old complete checkpoint or the new one.
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
