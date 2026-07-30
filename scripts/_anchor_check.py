from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def load_prepared_volume(
    path: Path,
    *,
    crop_size: int,
    output_size: int,
    num_phases: int,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"anchor volume was not found: {path}")
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "ZYX":
            raise ValueError(
                f"anchor TIF axes must be ZYX, got {series.axes!r}: {path}"
            )
        if series.dtype != np.dtype(np.uint8):
            raise ValueError(
                f"anchor TIF must contain uint8 labels, got "
                f"{series.dtype}: {path}"
            )
        volume = np.asarray(series.asarray())

    if volume.ndim != 3 or any(size < crop_size for size in volume.shape):
        raise ValueError(
            f"anchor volume must be 3D and at least {crop_size} voxels per axis."
        )
    if volume.size == 0 or int(volume.min()) < 0:
        raise ValueError("anchor volume must not be empty or negative.")
    if int(volume.max()) >= num_phases:
        raise ValueError(
            f"anchor volume must contain labels from 0 to {num_phases - 1}."
        )

    starts = tuple((size - crop_size) // 2 for size in volume.shape)
    selection = tuple(
        slice(start, start + crop_size)
        for start in starts
    )
    labels = torch.from_numpy(np.array(volume[selection], copy=True)).long()
    if crop_size != output_size:
        labels = F.interpolate(
            labels[None, None].to(torch.float32),
            size=(output_size, output_size, output_size),
            mode="nearest",
        )[0, 0].to(torch.long)
    return labels, starts


def axis_slices(volume: torch.Tensor, axis: int) -> torch.Tensor:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return volume.movedim(axis, 0)


def phase_scores(
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
