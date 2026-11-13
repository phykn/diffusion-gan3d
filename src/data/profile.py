from collections.abc import Mapping

import numpy as np
import torch

from src.data.images import list_side_images
from src.data.phases import rebin_profile


def measure_depth_profile(planes, num_phases, num_bins):
    images = list_side_images(planes)
    if len({len(image) for image in images}) != 1:
        return None
    row_fractions = np.mean(
        [[(image == phase).mean(1) for phase in range(num_phases)] for image in images],
        axis=0,
    )
    bins = rebin_profile(torch.from_numpy(row_fractions)[None], num_bins)[0].T
    points = [
        [index / num_bins, round_fractions(fractions)]
        for index, fractions in enumerate(bins.tolist())
    ]
    points.append([1.0, points[-1][1]])
    return {"axis": "z", "interpolation": "constant", "points": points}


def round_fractions(fractions):
    rounded = [round(value, 4) for value in fractions]
    # Keep the rounded fractions on the simplex that vf_profile requires.
    largest = rounded.index(max(rounded))
    rounded[largest] = round(1.0 - sum(rounded) + rounded[largest], 4)
    return rounded


def validate_profile(spec, phases):
    keys = {"axis", "points", "interpolation"}
    if not isinstance(spec, Mapping) or set(spec) != keys:
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
        knots = torch.tensor([position for position, _ in points], dtype=torch.float64)
        values = torch.tensor([list(row) for _, row in points], dtype=torch.float64)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("invalid profile points.") from exc
    invalid = ValueError(
        "profile points must span 0..1 with increasing knots and phase simplex values."
    )
    if knots.ndim != 1 or len(knots) < 2 or values.shape != (len(knots), phases):
        raise invalid
    if not (torch.isfinite(knots).all() and torch.isfinite(values).all()):
        raise invalid
    if knots[0] != 0 or knots[-1] != 1 or (knots.diff() <= 0).any():
        raise invalid
    if (values < 0).any() or (values > 1).any():
        raise invalid
    if not torch.allclose(values.sum(1), torch.ones_like(knots), atol=1e-6, rtol=0):
        raise invalid
    return knots, values


def integrate_profile(
    knots, values, interpolation, depth, origin, spacing, extent, device
):
    edges = (origin + torch.arange(depth + 1, dtype=torch.float64) * spacing) / extent
    # The profile continues flat above 0 and below 1.
    integral = edges.clamp_max(0)[:, None] * values[0]
    integral += (edges.clamp_min(1) - 1)[:, None] * values[-1]
    for index in range(len(knots) - 1):
        start, stop = knots[index], knots[index + 1]
        covered = (edges.clamp(start, stop) - start)[:, None]
        integral += covered * values[index]
        if interpolation == "linear":
            slope = (values[index + 1] - values[index]) / (stop - start)
            integral += covered.square() * slope / 2
    cell_fractions = integral.diff(dim=0) / (spacing / extent)
    return cell_fractions.T[None].to(device=device, dtype=torch.float32)
