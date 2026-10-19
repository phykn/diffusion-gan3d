import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn


def atomic_torch_save(payload, path: str | Path) -> Path:
    """Replace path with a complete artifact; readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary_path = Path(temporary.name)
    complete = False
    retain = False
    try:
        with temporary:
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        complete = True
        _retry_file_operation(lambda: temporary_path.replace(path))
    except OSError as error:
        if complete and getattr(error, "winerror", None) in (5, 32, 33):
            retain = True
            error.add_note(
                f"Publication failed; complete file retained at "
                f"{temporary_path.resolve()}"
            )
        raise
    finally:
        if not retain:
            try:
                _retry_file_operation(lambda: temporary_path.unlink(missing_ok=True))
            except OSError:
                # Cleanup must not mask a save error or invalidate a published file.
                logging.getLogger(__name__).warning(
                    "Could not remove temporary file %s",
                    temporary_path,
                    exc_info=True,
                )
    return path


def _retry_file_operation(operation: Callable[[], object]) -> None:
    for attempt in range(11):
        try:
            operation()
            return
        except OSError as error:
            # Windows can report a locked replacement target as access denied (5).
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 10:
                raise
            time.sleep(min(0.1 * 2**attempt, 1.0))


def torch_save(payload, path: str | Path, *, overwrite: bool = True) -> Path:
    """Write directly; callers publishing new files must track completion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("wb" if overwrite else "xb")
    try:
        with file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        if not overwrite:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning(
                    "Could not remove incomplete checkpoint %s", path, exc_info=True
                )
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
    return torch_save(model.state_dict(), path)


def load_model(path: str | Path, model: nn.Module) -> nn.Module:
    state = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model
