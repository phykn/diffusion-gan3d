import torch
import torch.nn.functional as F

from src.device import adaptive_pool, assert_all, wide_dtype


def encode_phases(labels, num_phases):
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("phase labels must have an integer dtype.")
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
    source = tuple(probs.shape[2:])
    if source == shape:
        return probs
    reduced = tuple(min(new, old) for new, old in zip(shape, source, strict=True))
    if reduced != source:
        if all(old % new == 0 for old, new in zip(source, reduced)):
            probs = adaptive_pool(probs, reduced)
        else:
            for axis, (old, size) in enumerate(zip(source, reduced), 2):
                if size != old:
                    probs = resize_axis(probs, axis, size)
    if reduced == shape:
        return probs
    for axis, size in enumerate(shape, 2):
        if size % probs.shape[axis] != 0:
            probs = resize_axis(probs, axis, size)
    mode = "trilinear" if probs.ndim == 5 else "bilinear"
    return F.interpolate(probs, size=shape, mode=mode, align_corners=False)


def resize_axis(probs, axis, size):
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


def measure_image_profile(images, bins):
    row_fractions = images.float().mean(dim=3)
    return rebin_profile(row_fractions, bins)


def rebin_profile(values, size):
    values = values.float()
    if size == values.shape[-1]:
        return values
    return resize_axis(values, -1, size)


def downsample(probs, shape):
    if any(new > old for new, old in zip(shape, probs.shape[2:])):
        raise ValueError("downsample cannot expand spatial dimensions.")
    return resize_phases(probs.float(), shape)


def upsample_region(probs, start, shape, scale):
    if (
        type(scale) is not int
        or scale < 1
        or any(s % scale or n < 1 or n % scale for s, n in zip(start, shape))
    ):
        raise ValueError("coarse regions must align with the integer LR/HR lattice.")
    source, padding = [], []
    for s, n, total in zip(start, shape, probs.shape[-3:]):
        left, right = s // scale - 1, (s + n) // scale + 1
        source.append(slice(max(0, left), min(total, right)))
        padding.append((max(0, -left), max(0, right - total)))
    low = probs[(Ellipsis, *source)]
    low = F.pad(
        low, tuple(v for pair in reversed(padding) for v in pair), mode="replicate"
    )
    high = F.interpolate(low, scale_factor=scale, mode="trilinear", align_corners=False)
    return high[(Ellipsis, *(slice(scale, scale + n) for n in shape))]
