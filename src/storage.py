import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import tifffile
import torch


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def torch_save(payload, path, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    partial = Path(file.name)
    try:
        with file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            partial.replace(path)
        else:
            try:
                os.link(partial, path)
            except FileExistsError:
                raise FileExistsError(f"{path} already exists.") from None
    finally:
        partial.unlink(missing_ok=True)
    return path


def load_volume(path):
    values = np.asarray(tifffile.imread(Path(path)))
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("volume must be a 3D TIFF with integer phase labels.")
    return torch.from_numpy(values.astype(np.int64, copy=True))


def save_volume(volume, path):
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


def validate_probabilities(probs):
    if (
        not isinstance(probs, torch.Tensor)
        or probs.ndim != 4
        or probs.numel() == 0
        or not probs.dtype.is_floating_point
    ):
        raise ValueError("fractional volume must be a C,D,H,W floating tensor.")
    return probs


def load_probabilities(path):
    probs = torch.load(path, map_location="cpu", weights_only=True)
    return validate_probabilities(probs)


def save_probabilities(probs, path):
    probs = validate_probabilities(probs)
    return torch_save(probs.detach().float().cpu(), path)
