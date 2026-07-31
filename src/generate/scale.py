import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from warnings import warn

import torch
from tqdm import tqdm

from ..anchor import PlaneAnchor
from .sample import Sampler


@dataclass(frozen=True)
class ScaleStats:
    shape: tuple[int, int, int]
    block_grid: tuple[int, int, int]
    block_size: int
    overlap: int
    block_count: int
    anchor_planes: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@torch.no_grad()
def generate_scaled(
    sampler: Sampler,
    *,
    blocks: int | Sequence[int],
    overlap: int | None = None,
    progress: bool = True,
) -> tuple[torch.Tensor, ScaleStats]:
    grid = _parse_grid(blocks)
    block_size = sampler.patch_size
    overlap = block_size // 2 if overlap is None else overlap
    if (
        not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or not 1 <= overlap <= block_size // 2
    ):
        raise ValueError("overlap must be between 1 and half the block size.")
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean.")

    stride = block_size - overlap
    starts = tuple(tuple(index * stride for index in range(count)) for count in grid)
    shape = tuple(block_size + (count - 1) * stride for count in grid)
    indices = tuple(product(*(range(count) for count in grid)))
    if len(indices) > 1 and not sampler.anchor_enabled:
        raise ValueError("scaled generation requires anchor-aware weights.")
    required_planes = sum(count > 1 for count in grid)
    if required_planes > sampler.max_anchor_planes:
        warn(
            "scale-up uses more simultaneous anchor axes than the weights "
            f"were trained for ({required_planes} > {sampler.max_anchor_planes}).",
            stacklevel=2,
        )

    scores = torch.zeros(
        sampler.num_phases,
        *shape,
        dtype=torch.float32,
    )
    anchor_planes = 0
    bar = tqdm(
        indices,
        total=len(indices),
        desc="Scale up",
        disable=not progress,
    )
    for index in bar:
        anchors = _make_anchors(
            index,
            starts=starts,
            scores=scores,
            block_size=block_size,
        )
        probabilities = sampler.sample(
            size=block_size,
            anchors=anchors,
        )
        anchor_planes += len(anchors)
        _blend(
            scores,
            probabilities,
            index=index,
            starts=starts,
            grid=grid,
            overlap=overlap,
        )

    labels = scores.argmax(dim=0).to(torch.uint8)
    stats = ScaleStats(
        shape=shape,
        block_grid=grid,
        block_size=block_size,
        overlap=overlap,
        block_count=len(indices),
        anchor_planes=anchor_planes,
        seams=tuple(
            tuple(cell[1] for cell in _split_axis(values, block_size, length)[:-1])
            for values, length in zip(starts, shape, strict=True)
        ),
    )
    return labels, stats


def _parse_grid(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        grid = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        grid = tuple(value)
    else:
        raise TypeError("blocks must be an integer or a sequence of three integers.")
    if len(grid) != 3 or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 1
        for count in grid
    ):
        raise ValueError("blocks must contain exactly three positive integers.")
    return grid


def _make_anchors(
    index: tuple[int, int, int],
    *,
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    scores: torch.Tensor,
    block_size: int,
) -> tuple[PlaneAnchor, ...]:
    origin = tuple(starts[axis][index[axis]] for axis in range(3))
    anchors = []
    for axis in range(3):
        if index[axis] == 0:
            continue
        previous_origin = starts[axis][index[axis] - 1]
        current_origin = origin[axis]
        actual_overlap = block_size - (current_origin - previous_origin)
        seam = current_origin + actual_overlap // 2
        target_index = seam - current_origin
        region = tuple(
            seam
            if source_axis == axis
            else slice(origin[source_axis], origin[source_axis] + block_size)
            for source_axis in range(3)
        )
        source_scores = scores[(slice(None), *region)]
        if bool((source_scores.sum(dim=0) <= 0.0).any().item()):
            raise RuntimeError("scale-up anchor region has not been generated.")
        # Global scores make overlapping anchor patches agree at shared coordinates.
        source = source_scores.argmax(dim=0)
        anchors.append(
            PlaneAnchor(
                labels=source.to(dtype=torch.long).clone(),
                axis=axis,
                index=target_index,
            )
        )
    return tuple(anchors)


def _blend(
    scores: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    index: tuple[int, int, int],
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    grid: tuple[int, int, int],
    overlap: int,
) -> None:
    block_size = probabilities.shape[1]
    weight = _make_weight(
        index,
        grid=grid,
        block_size=block_size,
        overlap=overlap,
        dtype=probabilities.dtype,
    )
    region = tuple(
        slice(starts[axis][index[axis]], starts[axis][index[axis]] + block_size)
        for axis in range(3)
    )
    scores[(slice(None), *region)].add_(probabilities * weight)


def _make_weight(
    index: tuple[int, int, int],
    *,
    grid: tuple[int, int, int],
    block_size: int,
    overlap: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(overlap, dtype=dtype)
    ramp = torch.sin((positions + 0.5) * math.pi / (2.0 * overlap)).square()
    weight = torch.ones((block_size,) * 3, dtype=dtype)
    for axis in range(3):
        axis_weight = torch.ones(block_size, dtype=dtype)
        if index[axis] > 0:
            axis_weight[:overlap] = ramp
        if index[axis] + 1 < grid[axis]:
            axis_weight[-overlap:] = ramp.flip(0)
        shape = [1, 1, 1]
        shape[axis] = block_size
        weight.mul_(axis_weight.reshape(shape))
    return weight


def _split_axis(
    starts: tuple[int, ...],
    block_size: int,
    size: int,
) -> tuple[tuple[int, int], ...]:
    seams = tuple((left + right + block_size) // 2 for left, right in pairwise(starts))
    boundaries = (0, *seams, size)
    return tuple(pairwise(boundaries))
