from pathlib import Path

from torch import nn

from ..data import AXES
from ..misc import atomic_torch_save

MODEL_WEIGHTS_NAME = "model.pt"


def save_model_weights(
    run_dir: str | Path,
    model: nn.Module,
    critics: nn.ModuleDict,
) -> Path:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module.")
    if not isinstance(critics, nn.ModuleDict):
        raise TypeError("critics must be a torch ModuleDict.")
    if set(critics) != {str(axis) for axis in AXES}:
        raise ValueError("critics must contain axes 0, 1, and 2.")

    root = Path(run_dir)
    path = root / MODEL_WEIGHTS_NAME
    atomic_torch_save(model.state_dict(), path)
    for axis in AXES:
        critic_path = root / f"critic_{axis}.pt"
        atomic_torch_save(critics[str(axis)].state_dict(), critic_path)
    return path
