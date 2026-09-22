import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn


def atomic_torch_save(payload, path: str | Path) -> Path:
    """Replace an artifact only after its complete serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_volume(path: str | Path) -> torch.Tensor:
    values = np.asarray(tifffile.imread(Path(path)))
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("volume must be a 3D TIFF with integer phase labels.")
    return torch.from_numpy(np.array(values, copy=True)).to(torch.long)


def save_volume(volume: torch.Tensor, path: str | Path) -> Path:
    if (
        volume.ndim != 3
        or volume.numel() == 0
        or volume.dtype.is_floating_point
        or volume.dtype.is_complex
    ):
        raise ValueError("volume must contain 3D integer phase labels.")
    if volume.dtype not in (torch.uint8, torch.bool):
        for slab in volume:
            labels = slab.to(torch.int64)
            if labels.min().item() < 0 or labels.max().item() > 255:
                raise ValueError(
                    "phase labels must be in [0, 255] for uint8 TIFF output."
                )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = volume.detach().to(device="cpu", dtype=torch.uint8)
    tifffile.imwrite(path, values.numpy())
    return path


def load_probabilities(path: str | Path) -> torch.Tensor:
    probs = torch.load(path, map_location="cpu", weights_only=True)
    return validate_probabilities(probs)


def validate_probabilities(probs: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(probs, torch.Tensor)
        or probs.ndim != 4
        or probs.numel() == 0
        or not probs.dtype.is_floating_point
    ):
        raise ValueError("fractional volume must be a C,D,H,W floating tensor.")
    return probs


def save_probabilities(probs: torch.Tensor, path: str | Path) -> Path:
    probs = validate_probabilities(probs)
    return atomic_torch_save(probs.detach().float().cpu(), path)


def save_model(path: str | Path, model: nn.Module) -> Path:
    return atomic_torch_save(model.state_dict(), path)


def load_model(path: str | Path, model: nn.Module) -> nn.Module:
    state = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model
