import math

import torch


def height_field(shape, axis, origin, pixel_size, extent, device=None):
    """Cell-center height in [-1, 1]; origin and extent use source-image pixels."""
    origin = torch.as_tensor(origin, device=device, dtype=torch.float32).reshape(-1, 1)
    extent = torch.as_tensor(extent, device=device, dtype=torch.float32).reshape(-1, 1)
    if not bool(torch.isfinite(extent).all() & (extent > 0).all()):
        raise ValueError("height extent must be finite and positive.")
    origin, extent = torch.broadcast_tensors(origin, extent)
    values = origin + (torch.arange(shape[axis], device=device) + 0.5) * pixel_size
    values = (2 * values / extent - 1).clamp(-1, 1)
    dims = [len(origin), 1] + [1] * len(shape)
    dims[axis + 2] = shape[axis]
    return values.reshape(dims).expand(len(origin), 1, *shape)


def resolve_extent(data, domain, extent=None):
    if extent is None:
        extent = data["height_extents"][domain]
    if extent is None:
        raise ValueError(
            "height_extent is required when source images have different heights."
        )
    if (
        isinstance(extent, bool)
        or not isinstance(extent, (int, float))
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("height_extent must be finite and positive (source pixels).")
    return float(extent)
