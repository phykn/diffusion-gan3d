import hashlib
import os
from pathlib import Path

import numpy as np
import tifffile
import torch


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def torch_save(payload, path: str | Path, *, overwrite: bool = True) -> Path:
    """Write to a temporary file, fsync, then rename it into place.

    Readers never see a partly written file. overwrite=False refuses an
    existing path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists.")
    partial = path.with_name(f".{path.name}.partial")
    try:
        with partial.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
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


def validate_probabilities(probs: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(probs, torch.Tensor)
        or probs.ndim != 4
        or probs.numel() == 0
        or not probs.dtype.is_floating_point
    ):
        raise ValueError("fractional volume must be a C,D,H,W floating tensor.")
    return probs


def load_probabilities(path: str | Path) -> torch.Tensor:
    probs = torch.load(path, map_location="cpu", weights_only=True)
    return validate_probabilities(probs)


def save_probabilities(probs: torch.Tensor, path: str | Path) -> Path:
    probs = validate_probabilities(probs)
    return torch_save(probs.detach().float().cpu(), path)
