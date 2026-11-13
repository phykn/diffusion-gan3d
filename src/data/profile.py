from collections.abc import Mapping

import numpy as np
import torch

from src.data.height import measure_height
from src.data.images import list_side_images
from src.data.resize import rebin_profile

PROFILE_KEYS = {"axis", "points", "interpolation"}
INTERPOLATIONS = ("linear", "constant")
ROUND_DIGITS = 4
SUM_TOLERANCE = 1e-6


def measure_depth_profile(planes, num_phases, num_bins):
    if measure_height(planes) is None:
        return None
    row_fractions = np.mean(
        [
            [(image == phase).mean(1) for phase in range(num_phases)]
            for image in list_side_images(planes)
        ],
        axis=0,
    )
    bin_fractions = rebin_profile(torch.from_numpy(row_fractions)[None], num_bins)[0].T
    points = [
        [index / num_bins, _round_fractions(fractions)]
        for index, fractions in enumerate(bin_fractions.tolist())
    ]
    points.append([1.0, points[-1][1]])
    return {"axis": "z", "interpolation": "constant", "points": points}


def measure_image_profile(images, bins):
    row_fractions = images.float().mean(dim=3)
    return rebin_profile(row_fractions, bins)


def validate_profile(spec, num_phases):
    if not isinstance(spec, Mapping) or set(spec) != PROFILE_KEYS:
        raise ValueError("vf_profile requires axis, points and interpolation.")
    if spec["axis"] != "z" or spec["interpolation"] not in INTERPOLATIONS:
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
        positions = torch.tensor([point[0] for point in points], dtype=torch.float64)
        fractions = torch.tensor(
            [list(point[1]) for point in points], dtype=torch.float64
        )
    except (TypeError, ValueError, IndexError, OverflowError) as error:
        raise ValueError("invalid profile points.") from error
    invalid = ValueError(
        "profile points must span 0..1 with increasing knots and phase simplex values."
    )
    count = len(positions)
    if positions.ndim != 1 or count < 2 or fractions.shape != (count, num_phases):
        raise invalid
    if not (torch.isfinite(positions).all() and torch.isfinite(fractions).all()):
        raise invalid
    if positions[0] != 0 or positions[-1] != 1 or (positions.diff() <= 0).any():
        raise invalid
    if (fractions < 0).any() or (fractions > 1).any():
        raise invalid
    sums = fractions.sum(1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=SUM_TOLERANCE, rtol=0):
        raise invalid
    return positions, fractions


def integrate_profile(
    positions,
    fractions,
    interpolation,
    depth,
    height_origin,
    pixels_per_cell,
    height_extent,
    device,
):
    cell_rows = torch.arange(depth + 1, dtype=torch.float64) * pixels_per_cell
    cell_edges = (height_origin + cell_rows) / height_extent
    # The profile continues flat above 0 and below 1.
    area = cell_edges.clamp_max(0)[:, None] * fractions[0]
    area += (cell_edges.clamp_min(1) - 1)[:, None] * fractions[-1]
    for index in range(len(positions) - 1):
        start, stop = positions[index], positions[index + 1]
        covered = (cell_edges.clamp(start, stop) - start)[:, None]
        area += covered * fractions[index]
        if interpolation == "linear":
            slope = (fractions[index + 1] - fractions[index]) / (stop - start)
            area += covered.square() * slope / 2
    cell_fractions = area.diff(dim=0) / (pixels_per_cell / height_extent)
    return cell_fractions.T[None].to(device=device, dtype=torch.float32)


def _round_fractions(fractions):
    rounded = [round(value, ROUND_DIGITS) for value in fractions]
    # Keep the rounded fractions on the simplex that vf_profile requires.
    largest = rounded.index(max(rounded))
    rounded[largest] = round(1.0 - sum(rounded) + rounded[largest], ROUND_DIGITS)
    return rounded
