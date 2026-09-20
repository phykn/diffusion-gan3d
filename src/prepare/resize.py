import math

import torch
import torch.nn.functional as F


def scaled_size(size: int, scale: float) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale < 1
    ):
        raise ValueError("scale_factor must be finite and at least 1.")
    value = size * scale
    if not math.isfinite(value) or not math.isclose(
        value, round(value), abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError(
            "size * scale_factor must be an integer; sizes are never rounded."
        )
    return round(value)


def phase_channels(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("phase labels must have an integer dtype.")
    if labels.numel() == 0:
        raise ValueError(f"phase labels must be in [0, {num_phases - 1}].")
    indices = labels.to(torch.int64)
    valid = ((indices >= 0) & (indices < num_phases)).all()
    if labels.device.type == "cuda":
        torch._assert_async(valid, "phase labels are outside num_phases.")
    elif not bool(valid):
        raise ValueError(f"phase labels must be in [0, {num_phases - 1}].")
    channels = torch.zeros(
        (labels.shape[0], num_phases, *labels.shape[1:]),
        device=labels.device,
        dtype=torch.float32,
    )
    return channels.scatter_(1, indices.unsqueeze(1), 1.0)


def resize_phases(probs: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
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
        probs = F.interpolate(probs, size=reduced, mode="area")
    if reduced == shape:
        return probs
    mode = "trilinear" if probs.ndim == 5 else "bilinear"
    return F.interpolate(probs, size=shape, mode=mode, align_corners=False)


def resize_labels(labels: torch.Tensor, size: int, num_phases: int) -> torch.Tensor:
    return resize_crop(labels, size, num_phases).argmax(0)


def resize_crop(
    labels: torch.Tensor,
    size: int,
    num_phases: int,
) -> torch.Tensor:
    probs = phase_channels(labels.unsqueeze(0), num_phases)
    return resize_phases(probs, (size,) * labels.ndim).squeeze(0)


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
