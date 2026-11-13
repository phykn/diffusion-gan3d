import math

import torch

from src.data.images import list_side_images


def measure_height(planes):
    sizes = {image.shape[0] for image in list_side_images(planes)}
    return sizes.pop() if len(sizes) == 1 else None


def check_extent(extent):
    if (
        isinstance(extent, bool)
        or not isinstance(extent, (int, float))
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("height_extent must be finite and positive (source pixels).")
    return float(extent)


def build_heights(shape, origin, spacing, extent, device):
    origin = torch.as_tensor(origin, device=device, dtype=torch.float32).reshape(-1, 1)
    extent = torch.as_tensor(extent, device=device, dtype=torch.float32).reshape(-1, 1)
    values = origin + (torch.arange(shape[0], device=device) + 0.5) * spacing
    values = (2 * values / extent - 1).clamp(-1, 1)
    dims = (len(origin), 1, shape[0], *((1,) * (len(shape) - 1)))
    return values.reshape(dims).expand(len(origin), 1, *shape)
