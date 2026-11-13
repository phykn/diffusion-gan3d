from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.device import adaptive_pool, assert_all, wide_dtype


def phase_channels(labels, num_phases):
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


def resize_phases(probs, shape):
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
    if reduced != source:
        if all(old % new == 0 for old, new in zip(source, reduced, strict=True)):
            probs = adaptive_pool(probs, reduced)
        else:
            for axis, (old, size) in enumerate(zip(source, reduced, strict=True), 2):
                if size != old:
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


def resize_anchor(labels, size, num_phases):
    probs = phase_channels(labels.unsqueeze(0), num_phases)
    probs = resize_phases(probs, (size,) * labels.ndim).squeeze(0)
    # Area resizing can round a full cell to 1 + eps; anchors require [0, 1].
    return probs.clamp_(0.0, 1.0)


def downsample(probs, shape):
    if any(new > old for new, old in zip(shape, probs.shape[2:], strict=True)):
        raise ValueError("downsample cannot expand spatial dimensions.")
    return resize_phases(probs.float(), shape)


def coarse_region(probs, start, shape, scale):
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
    low = probs[(Ellipsis, *source)]
    low = F.pad(
        low, tuple(v for pair in reversed(padding) for v in pair), mode="replicate"
    )
    high = F.interpolate(low, scale_factor=scale, mode="trilinear", align_corners=False)
    return high[(Ellipsis, *(slice(scale, scale + n) for n in shape))]


@dataclass(frozen=True)
class Crops:
    images: torch.Tensor  # [B, K, patch, patch] phase fractions
    origins: torch.Tensor  # [B] crop top row in source pixels (z for side planes)
    extents: torch.Tensor  # [B] source image heights in pixels


class CropSampler:
    def __init__(self, images, crop_size, patch_size, num_phases):
        self.groups = images.images
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

    def sample(self, count, generator):
        group = self.groups[
            int(torch.randint(len(self.groups), (), generator=generator))
        ]
        choices = torch.randint(len(group), (count,), generator=generator).tolist()
        size = self.crop_size
        crops, origins, extents = [], [], []
        for choice in choices:
            image = group[choice]
            height, width = image.shape
            top = int(torch.randint(height - size + 1, (), generator=generator))
            left = int(torch.randint(width - size + 1, (), generator=generator))
            crops.append(image[top : top + size, left : left + size])
            origins.append(float(top))
            extents.append(float(height))
        labels = torch.from_numpy(np.stack(crops))
        probs = phase_channels(labels, self.num_phases)
        images = resize_phases(probs, (self.patch_size,) * 2)
        return Crops(
            images=images,
            origins=torch.tensor(origins),
            extents=torch.tensor(extents),
        )
