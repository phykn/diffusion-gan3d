import math

import torch

from src.data.images import list_side_images

TOP, BOTTOM = -1.0, 1.0


def measure_height(planes):
    heights = {image.shape[0] for image in list_side_images(planes)}
    return heights.pop() if len(heights) == 1 else None


def check_extent(extent):
    if (
        isinstance(extent, bool)
        or not isinstance(extent, (int, float))
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("height_extent must be finite and positive (source pixels).")
    return float(extent)


def build_heights(shape, height_origin, pixels_per_cell, height_extent, device):
    origin = torch.as_tensor(height_origin, device=device, dtype=torch.float32)
    extent = torch.as_tensor(height_extent, device=device, dtype=torch.float32)
    origin, extent = origin.reshape(-1, 1), extent.reshape(-1, 1)
    batch, depth = len(origin), shape[0]
    cell_centers = torch.arange(depth, device=device) + 0.5
    source_rows = origin + cell_centers * pixels_per_cell
    heights = (2 * source_rows / extent - 1).clamp(TOP, BOTTOM)
    flat_dims = [1] * (len(shape) - 1)
    return heights.reshape(batch, 1, depth, *flat_dims).expand(batch, 1, *shape)
