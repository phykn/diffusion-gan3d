from pathlib import Path

import numpy as np
import tifffile
import torch


def load_volume(
    path: str | Path,
    *,
    shape: tuple[int, int, int] | None = None,
    num_phases: int | None = None,
) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label volume was not found: {path}")
    values = np.asarray(tifffile.imread(path))
    if values.ndim != 3 or values.size == 0:
        raise ValueError("label volume must be a non-empty 3D array.")
    if values.dtype != np.uint8:
        raise ValueError(
            f"label volume must contain uint8 phases, got {values.dtype}: {path}"
        )
    if shape is not None and values.shape != shape:
        raise ValueError(f"label volume must have shape {shape}, got {values.shape}.")
    if num_phases is not None:
        if (
            not isinstance(num_phases, int)
            or isinstance(num_phases, bool)
            or not 1 <= num_phases <= 256
        ):
            raise ValueError("num_phases must be an integer from 1 to 256.")
        if int(values.max()) >= num_phases:
            raise ValueError(
                f"label volume must contain phases from 0 to {num_phases - 1}."
            )
    return torch.from_numpy(np.array(values, copy=True)).to(torch.long)


def save_volume(volume: torch.Tensor, path: str | Path) -> None:
    if not isinstance(volume, torch.Tensor) or volume.ndim != 3 or volume.numel() == 0:
        raise ValueError(
            "label volume must be a non-empty tensor with shape [D, H, W]."
        )
    if volume.is_floating_point() or volume.is_complex() or volume.dtype == torch.bool:
        raise TypeError("label volume must use an integer dtype.")
    values = volume.detach().to(device="cpu")
    lower, upper = torch.aminmax(values)
    if int(lower) < 0 or int(upper) > 255:
        raise ValueError("label volume phases must be between 0 and 255.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, values.to(torch.uint8).numpy())
