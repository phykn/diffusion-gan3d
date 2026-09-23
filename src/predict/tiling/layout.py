import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise, product

import torch


@dataclass(frozen=True)
class TilePlan:
    """Tiling geometry in z/y/x order, with the padded grid kept for reporting."""

    shape: tuple[int, int, int]
    tile_size: int
    overlap: int
    stride: int
    grid: tuple[int, int, int]
    tile_count: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    generation_shape: tuple[int, int, int] | None = None
    margin: int = 0
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        # Preserve direct construction while keeping starts in generation space,
        # including when output_plan replaces shape/seams for reporting.
        if self.starts is None:
            shape = self.generation_shape or self.shape
            object.__setattr__(
                self,
                "starts",
                tuple(
                    axis_starts(size, min(size, self.tile_size), self.stride)
                    for size in shape
                ),
            )

    @property
    def base_shell(self) -> int:
        return min(self.overlap // 2, (self.stride - 1) // 2)


@dataclass(frozen=True)
class Tile:
    """Read overlapping context from source; write only the owned target region."""

    source: tuple[slice, slice, slice]
    target: tuple[slice, slice, slice]
    margins: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def make_plan(
    shape: tuple[int, int, int], tile_size: int, overlap: int
) -> TilePlan:
    """Build grid geometry from validated generation-space dimensions."""
    stride = tile_size - 2 * overlap
    lengths = tuple(min(size, tile_size) for size in shape)
    starts = tuple(
        axis_starts(size, length, stride) for size, length in zip(shape, lengths)
    )
    grid = tuple(len(axis) for axis in starts)
    return TilePlan(
        shape=shape,
        tile_size=tile_size,
        overlap=overlap,
        stride=stride,
        grid=grid,
        tile_count=math.prod(grid),
        starts=starts,
        seams=tuple(
            tuple((left + length + right) // 2 for left, right in pairwise(axis))
            for axis, length in zip(starts, lengths)
        ),
    )


def output_plan(
    plan: TilePlan,
    output_shape: tuple[int, int, int],
) -> TilePlan:
    """Report output-space shape/seams while retaining the generation tile grid."""
    margin = plan.margin
    seams = tuple(
        tuple(
            seam - margin for seam in axis if margin < seam < plan.shape[index] - margin
        )
        for index, axis in enumerate(plan.seams)
    )
    return replace(
        plan,
        shape=output_shape,
        seams=seams,
        generation_shape=plan.generation_shape or plan.shape,
    )


def crop_output(
    volume: torch.Tensor,
    output_shape: tuple[int, int, int],
    margin: int,
) -> torch.Tensor:
    if margin == 0:
        return volume
    region = tuple(slice(margin, margin + size) for size in output_shape)
    leading = (slice(None),) * (volume.ndim - 3)
    return volume[leading + region].clone()


def axis_starts(size: int, tile_size: int, stride: int) -> tuple[int, ...]:
    if size == tile_size:
        return (0,)
    count = math.ceil((size - tile_size) / stride) + 1
    starts = [index * stride for index in range(count - 1)]
    starts.append(size - tile_size)
    return tuple(starts)


def parse_shape(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        shape = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        shape = tuple(value)
    else:
        raise TypeError("shape must be an integer or a sequence of three integers.")
    if len(shape) != 3 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in shape
    ):
        raise ValueError("shape must contain exactly three positive integers.")
    return shape


def make_tiles(
    plan: TilePlan,
) -> tuple[Tile, ...]:
    if plan.generation_shape is not None and plan.shape != plan.generation_shape:
        raise ValueError("output-space statistics cannot be used as a generation plan.")
    lengths = tuple(min(size, plan.tile_size) for size in plan.shape)
    tiles = []
    for idx in product(
        range(plan.grid[0]),
        range(plan.grid[1]),
        range(plan.grid[2]),
    ):
        source = []
        target = []
        margins = []
        for axis, tile_idx in enumerate(idx):
            source_start = plan.starts[axis][tile_idx]
            source_stop = source_start + lengths[axis]
            target_start = 0 if tile_idx == 0 else plan.seams[axis][tile_idx - 1]
            target_stop = (
                plan.shape[axis]
                if tile_idx + 1 == plan.grid[axis]
                else plan.seams[axis][tile_idx]
            )
            source.append(slice(source_start, source_stop))
            target.append(slice(target_start, target_stop))
            margins.append(
                (
                    plan.overlap if tile_idx > 0 else 0,
                    plan.overlap if tile_idx + 1 < plan.grid[axis] else 0,
                )
            )
        tiles.append(
            Tile(
                source=tuple(source),
                target=tuple(target),
                margins=tuple(margins),
            )
        )
    return tuple(tiles)
