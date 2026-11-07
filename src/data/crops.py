"""Random crops of the label images, resized by area to the model grid."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.data.images import PlaneImages
from src.device import adaptive_pool, assert_all, wide_dtype


def phase_channels(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    """One-hot [B, *S] integer labels into float32 [B, K, *S]."""
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("phase labels must have an integer dtype.")
    if labels.numel() == 0:
        raise ValueError(f"phase labels must be in [0, {num_phases - 1}].")
    indices = labels.to(torch.int64)
    valid = ((indices >= 0) & (indices < num_phases)).all()
    assert_all(valid, f"phase labels must be in [0, {num_phases - 1}].")
    channels = torch.zeros(
        (labels.shape[0], num_phases, *labels.shape[1:]),
        device=labels.device,
        dtype=torch.float32,
    )
    return channels.scatter_(1, indices.unsqueeze(1), 1.0)


def resize_phases(probs: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Resize cell fractions without changing the amount of each phase.

    Integer reductions average-pool; other reductions integrate the exact
    source/target cell overlaps (PyTorch's area mode overlaps its bins there);
    enlargement interpolates linearly from cell centers.
    """
    shape = tuple(shape)
    if (
        probs.ndim not in (4, 5)
        or len(shape) != probs.ndim - 2
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError("phase channels require a positive 2D or 3D output shape.")
    source = tuple(probs.shape[2:])
    if source == shape:
        return probs
    reduced = tuple(min(new, old) for new, old in zip(shape, source, strict=True))
    if reduced != source and all(old % new == 0 for old, new in zip(source, reduced)):
        probs = adaptive_pool(probs, reduced)
    else:
        for axis, size in enumerate(reduced, 2):
            if size != probs.shape[axis]:
                probs = _resize_fraction_axis(probs, axis, size)
    if reduced == shape:
        return probs
    for axis, size in enumerate(shape, 2):
        if size % probs.shape[axis] != 0:
            probs = _resize_fraction_axis(probs, axis, size)
    mode = "trilinear" if probs.ndim == 5 else "bilinear"
    return F.interpolate(probs, size=shape, mode=mode, align_corners=False)


def _resize_fraction_axis(probs, axis, size):
    source = probs.shape[axis]
    dtype = wide_dtype(probs.device)
    edges = torch.arange(size + 1, device=probs.device, dtype=dtype)
    edges = edges * source / size
    cells = torch.arange(source, device=probs.device, dtype=dtype)
    overlap = (
        torch.minimum(edges[1:, None], cells + 1)
        - torch.maximum(edges[:-1, None], cells)
    ).clamp_min(0) / (source / size)
    # Keep conservation and simplex accuracy under training autocast as well.
    with torch.autocast(device_type=probs.device.type, enabled=False):
        values = probs if probs.dtype == torch.float64 else probs.float()
        return (values.movedim(axis, -1) @ overlap.to(values.dtype).T).movedim(-1, axis)


def resize_crop(labels: torch.Tensor, size: int, num_phases: int) -> torch.Tensor:
    """[*S] labels to [K, size, ...] fractions; ties stay fractional."""
    probs = phase_channels(labels.unsqueeze(0), num_phases)
    return resize_phases(probs, (size,) * labels.ndim).squeeze(0)


def resize_anchor(labels: torch.Tensor, size: int, num_phases: int) -> torch.Tensor:
    # Area resizing can round a full cell to 1 + eps; anchors require [0, 1].
    return resize_crop(labels, size, num_phases).clamp_(0.0, 1.0)


def downsample(probs: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if any(new > old for new, old in zip(shape, probs.shape[2:], strict=True)):
        raise ValueError("downsample cannot expand spatial dimensions.")
    return resize_phases(probs.float(), shape)


def coarse_region(
    probs: torch.Tensor,
    start: tuple[int, int, int],
    shape: tuple[int, int, int],
    scale: int,
) -> torch.Tensor:
    """HR-grid trilinear upsampling of an LR region with one LR cell of halo."""
    if (
        type(scale) is not int
        or scale < 1
        or any(
            s % scale or n < 1 or n % scale for s, n in zip(start, shape, strict=True)
        )
    ):
        raise ValueError("coarse regions must align with the integer LR/HR lattice.")
    source, padding = [], []
    for s, n, total in zip(start, shape, probs.shape[-3:], strict=True):
        left, right = s // scale - 1, (s + n) // scale + 1
        source.append(slice(max(0, left), min(total, right)))
        padding.append((max(0, -left), max(0, right - total)))
    low = probs[(slice(None), slice(None), *source)]
    low = F.pad(
        low, tuple(v for pair in reversed(padding) for v in pair), mode="replicate"
    )
    high = F.interpolate(low, scale_factor=scale, mode="trilinear", align_corners=False)
    return high[(slice(None), slice(None), *(slice(scale, scale + n) for n in shape))]


def scaled_size(size: int, scale: float) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    value = size * scale
    if not math.isfinite(value) or not math.isclose(
        value, round(value), abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError("size * scale must be an integer; sizes are never rounded.")
    return round(value)


@dataclass(frozen=True)
class Crops:
    """A batch of resized crops of one plane and where they came from."""

    images: torch.Tensor  # [B, K, patch, patch] phase fractions
    origins: torch.Tensor  # [B] crop top row in source pixels (z for side planes)
    extents: torch.Tensor  # [B] source image heights in pixels


class CropSampler:
    """Draw crops of one plane: a random folder, then random images and crops.

    Every choice comes from the generator passed in, so a seeded generator
    repeats the same crops.
    """

    def __init__(
        self, images: PlaneImages, crop_size: int, patch_size: int, num_phases: int
    ):
        self.images = images
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_phases = num_phases
        for group, paths in zip(images.images, images.paths, strict=True):
            for image, path in zip(group, paths, strict=True):
                if crop_size > min(image.shape):
                    raise ValueError(
                        f"crop_size {crop_size} does not fit inside {path} "
                        f"({image.shape[0]}x{image.shape[1]})."
                    )

    def sample(self, count: int, generator: torch.Generator) -> Crops:
        groups = self.images.images
        group = groups[int(torch.randint(len(groups), (), generator=generator))]
        choices = torch.randint(len(group), (count,), generator=generator).tolist()
        size = self.crop_size
        crops, origins, extents = [], [], []
        for choice in choices:
            image = group[choice]
            height, width = image.shape
            top = int(torch.randint(height - size + 1, (), generator=generator))
            left = int(torch.randint(width - size + 1, (), generator=generator))
            crop = np.array(image[top : top + size, left : left + size])
            crops.append(torch.from_numpy(crop).long())
            origins.append(float(top))
            extents.append(float(height))
        labels = torch.stack(crops)
        probs = phase_channels(labels, self.num_phases)
        images = resize_phases(probs, (self.patch_size,) * 2)
        return Crops(
            images=images,
            origins=torch.tensor(origins),
            extents=torch.tensor(extents),
        )
