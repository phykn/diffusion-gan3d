import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from src.axes import PLANES
from src.device import wide_dtype


def grid_spacing(crop_size, patch_size):
    return crop_size / patch_size


def height_field(shape, origin, spacing, extent, device=None):
    origin = torch.as_tensor(origin, device=device, dtype=torch.float32).reshape(-1, 1)
    extent = torch.as_tensor(extent, device=device, dtype=torch.float32).reshape(-1, 1)
    if not bool(torch.isfinite(extent).all() & (extent > 0).all()):
        raise ValueError("height extent must be finite and positive.")
    origin, extent = torch.broadcast_tensors(origin, extent)
    values = origin + (torch.arange(shape[0], device=device) + 0.5) * spacing
    values = (2 * values / extent - 1).clamp(-1, 1)
    dims = (len(origin), 1, shape[0], *((1,) * (len(shape) - 1)))
    return values.reshape(dims).expand(len(origin), 1, *shape)


def check_extent(extent):
    if (
        isinstance(extent, bool)
        or not isinstance(extent, (int, float))
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("height_extent must be finite and positive (source pixels).")
    return float(extent)


def side_images(planes):
    return [
        image
        for plane, images in planes.items()
        if plane != PLANES[0]
        for image in images.all_images()
    ]


def height_extent(planes):
    sizes = {image.shape[0] for image in side_images(planes)}
    return sizes.pop() if len(sizes) == 1 else None


def depth_profile(planes, num_phases, num_bins):
    images = side_images(planes)
    if len({len(image) for image in images}) != 1:
        return None
    rows = np.mean(
        [np.stack([(img == p).mean(1) for p in range(num_phases)]) for img in images],
        axis=0,
    )
    bands = rebin_profile(torch.from_numpy(rows)[None], num_bins)[0].T.numpy()
    values = []
    for band in bands:
        row = [round(float(v), 4) for v in band]
        # Keep the rounded fractions on the simplex that vf_profile requires.
        top = row.index(max(row))
        row[top] = round(1.0 - sum(row) + row[top], 4)
        values.append(row)
    values.append(values[-1])
    return {
        "axis": "z",
        "interpolation": "constant",
        "points": [[i / num_bins, row] for i, row in enumerate(values)],
    }


def rebin_profile(values, size):
    if type(size) is not int or size < 1 or values.ndim != 3 or values.numel() == 0:
        raise ValueError("profile requires non-empty [B,K,D] and a positive bin count.")
    values = values.float()
    depth = values.shape[-1]
    if size == depth:
        return values
    dtype = wide_dtype(values.device)
    edges = torch.linspace(0, depth, size + 1, device=values.device, dtype=dtype)
    indices = edges.long().clamp_max(depth - 1)
    # Double precision avoids cancellation when subtracting nearby large integrals.
    cumulative = F.pad(values.cumsum(-1, dtype=dtype), (1, 0))
    integral = cumulative.index_select(-1, indices) + values.index_select(
        -1, indices
    ) * (edges - indices)
    return (integral.diff(dim=-1) / (depth / size)).float()


def image_profile(images, bins):
    if images.ndim != 4 or images.numel() == 0:
        raise ValueError("images must have non-empty shape [B,K,H,W].")
    return rebin_profile(images.float().mean(dim=3), bins)


def validate_profile(spec, phases):
    if not isinstance(spec, Mapping) or set(spec) != {
        "axis",
        "points",
        "interpolation",
    }:
        raise ValueError("vf_profile requires axis, points and interpolation.")
    if spec["axis"] != "z" or spec["interpolation"] not in ("linear", "constant"):
        raise ValueError(
            "vf_profile requires z axis and linear or constant interpolation."
        )
    points = spec["points"]
    if not isinstance(points, (list, tuple)) or any(
        not isinstance(point, (list, tuple))
        or len(point) != 2
        or not isinstance(point[1], (list, tuple))
        for point in points
    ):
        raise ValueError("profile points must be [position, phase fractions] pairs.")
    try:
        knots = torch.tensor([p[0] for p in points], dtype=torch.float64)
        values = torch.tensor([list(p[1]) for p in points], dtype=torch.float64)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("invalid profile points.") from exc
    if (
        knots.ndim != 1
        or len(knots) < 2
        or values.shape != (len(knots), phases)
        or not bool(torch.isfinite(knots).all() & torch.isfinite(values).all())
        or knots[0] != 0
        or knots[-1] != 1
        or not bool((knots.diff() > 0).all())
        or not bool(((values >= 0) & (values <= 1)).all())
        or not torch.allclose(
            values.sum(1), torch.ones(len(knots), dtype=values.dtype), atol=1e-6, rtol=0
        )
    ):
        raise ValueError(
            "profile points must span 0..1 with increasing knots and phase simplex values."
        )
    return knots, values


def integrate_profile(
    knots, values, interpolation, depth, origin, spacing, extent, device
):
    edges = (origin + torch.arange(depth + 1, dtype=torch.float64) * spacing) / extent
    left, right = edges[:-1], edges[1:]
    result = torch.zeros(depth, values.shape[1], dtype=torch.float64)
    for index in range(len(knots) - 1):
        a = left.clamp(knots[index], knots[index + 1])
        b = right.clamp(knots[index], knots[index + 1])
        width = b - a
        result += width[:, None] * values[index]
        if interpolation == "linear":
            slope = (values[index + 1] - values[index]) / (
                knots[index + 1] - knots[index]
            )
            result += (
                ((b - knots[index]).square() - (a - knots[index]).square())[:, None]
                * slope
                / 2
            )
    result += (right.clamp_max(0) - left.clamp_max(0))[:, None] * values[0]
    result += (right.clamp_min(1) - left.clamp_min(1))[:, None] * values[-1]
    return (
        (result / (spacing / extent))
        .T.unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
