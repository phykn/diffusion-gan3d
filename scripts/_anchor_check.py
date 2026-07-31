from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def load_volume(
    path: Path,
    *,
    patch_size: int,
    num_phases: int,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"anchor volume was not found: {path}")
    volume = np.asarray(tifffile.imread(path))

    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("anchor volume must be a non-empty 3D array.")
    if volume.dtype != np.uint8:
        raise ValueError(
            f"anchor volume must contain uint8 labels, got {volume.dtype}: {path}"
        )
    if int(volume.max()) >= num_phases:
        raise ValueError(
            f"anchor volume must contain labels from 0 to {num_phases - 1}."
        )

    labels = torch.from_numpy(np.array(volume, copy=True)).long()
    if volume.shape != (patch_size, patch_size, patch_size):
        labels = F.interpolate(
            labels[None, None].to(torch.float32),
            size=(patch_size, patch_size, patch_size),
            mode="nearest",
        )[0, 0].to(torch.long)
    return labels


def slice_axis(volume: torch.Tensor, axis: int) -> torch.Tensor:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return volume.movedim(axis, 0)


def score_phases(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    num_phases: int,
) -> tuple[list[float], list[float]]:
    if generated.shape != target.shape:
        raise ValueError("generated and target labels must have the same shape.")
    iou = []
    recall = []
    for phase in range(num_phases):
        predicted = generated == phase
        expected = target == phase
        intersection = int((predicted & expected).sum())
        union = int((predicted | expected).sum())
        support = int(expected.sum())
        iou.append(1.0 if union == 0 else intersection / union)
        recall.append(1.0 if support == 0 else intersection / support)
    return iou, recall
