import torch
import torch.nn.functional as F

from src.device import adaptive_pool, wide_dtype


def resize_phases(fractions, shape):
    shape = tuple(shape)
    current = tuple(fractions.shape[2:])
    shrunk = tuple(min(new, old) for new, old in zip(shape, current, strict=True))
    if shrunk != current:
        fractions = _shrink(fractions, shrunk)
    if shrunk != shape:
        fractions = _grow(fractions, shape)
    return fractions


def downsample(fractions, shape):
    if any(new > old for new, old in zip(shape, fractions.shape[2:])):
        raise ValueError("downsample cannot expand spatial dimensions.")
    return resize_phases(fractions.float(), shape)


def upsample_region(low_fractions, start, shape, scale):
    if (
        type(scale) is not int
        or scale < 1
        or any(s % scale or n < 1 or n % scale for s, n in zip(start, shape))
    ):
        raise ValueError("coarse regions must align with the integer LR/HR lattice.")
    low_slices, padding = [], []
    for s, n, low_size in zip(start, shape, low_fractions.shape[-3:]):
        # One more LR cell on each side gives the interpolation its neighbours.
        first, last = s // scale - 1, (s + n) // scale + 1
        low_slices.append(slice(max(0, first), min(low_size, last)))
        padding.append((max(0, -first), max(0, last - low_size)))
    region = low_fractions[(Ellipsis, *low_slices)]
    # F.pad takes the last axis first.
    last_axis_first = tuple(v for pair in reversed(padding) for v in pair)
    region = F.pad(region, last_axis_first, mode="replicate")
    high = F.interpolate(
        region, scale_factor=scale, mode="trilinear", align_corners=False
    )
    return high[(Ellipsis, *(slice(scale, scale + n) for n in shape))]


def rebin_profile(profile, bins):
    profile = profile.float()
    if bins == profile.shape[-1]:
        return profile
    return _resize_axis(profile, -1, bins)


def _shrink(fractions, shape):
    current = tuple(fractions.shape[2:])
    if all(old % new == 0 for old, new in zip(current, shape)):
        return adaptive_pool(fractions, shape)
    for axis, (old, new) in enumerate(zip(current, shape), 2):
        if new != old:
            fractions = _resize_axis(fractions, axis, new)
    return fractions


def _grow(fractions, shape):
    for axis, size in enumerate(shape, 2):
        if size % fractions.shape[axis] != 0:
            fractions = _resize_axis(fractions, axis, size)
    mode = "trilinear" if fractions.ndim == 5 else "bilinear"
    return F.interpolate(fractions, size=shape, mode=mode, align_corners=False)


def _resize_axis(fractions, axis, size):
    old_size = fractions.shape[axis]
    device, dtype = fractions.device, wide_dtype(fractions.device)
    new_edges = torch.arange(size + 1, device=device, dtype=dtype) * old_size / size
    old_cells = torch.arange(old_size, device=device, dtype=dtype)
    overlap = (
        torch.minimum(new_edges[1:, None], old_cells + 1)
        - torch.maximum(new_edges[:-1, None], old_cells)
    ).clamp_min(0)
    weights = overlap / (old_size / size)
    # Keep conservation and simplex accuracy under training autocast as well.
    with torch.autocast(device_type=device.type, enabled=False):
        values = fractions if fractions.dtype == torch.float64 else fractions.float()
        return (values.movedim(axis, -1) @ weights.to(values.dtype).T).movedim(-1, axis)
