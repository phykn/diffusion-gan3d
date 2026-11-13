from typing import Annotated, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.data.height import measure_height
from src.data.images import list_side_images
from src.data.resize import rebin_profile

ROUND_DIGITS = 4
SUM_TOLERANCE = 1e-6
Number = Annotated[float, Field(strict=True)]


class ProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    axis: Literal["z"]
    interpolation: Literal["linear", "constant"]
    points: tuple[tuple[Number, tuple[Number, ...]], ...] = Field(min_length=2)


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
    try:
        points = ProfileSpec.model_validate(spec).points
    except ValidationError as error:
        problem = error.errors()[0]
        place = ".".join(map(str, problem["loc"]))
        raise ValueError(f"vf_profile {place}: {problem['msg']}") from None
    invalid = ValueError(
        "profile points must span 0..1 with increasing knots and phase simplex values."
    )
    if any(len(fractions) != num_phases for _, fractions in points):
        raise invalid
    positions = torch.tensor([position for position, _ in points], dtype=torch.float64)
    fractions = torch.tensor(
        [fractions for _, fractions in points], dtype=torch.float64
    )
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
