import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise, product

import torch


@dataclass(frozen=True)
class Tile:
    """Read context from source; own (write the posterior of) target."""

    source: tuple[slice, slice, slice]
    target: tuple[slice, slice, slice]
    margins: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(region.stop - region.start for region in self.source)


@dataclass(frozen=True)
class TilePlan:
    """Generation geometry in z/y/x order.

    The generated region is the output plus margin m on every outer face;
    tiles have edge E = patch + 2m (or the region, when it is shorter).
    """

    output_shape: tuple[int, int, int]
    margin: int
    overlap: int
    tile_edge: tuple[int, int, int]
    shape: tuple[int, int, int]
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

    @property
    def tile_count(self) -> int:
        return math.prod(self.grid)

    @cached_property
    def tiles(self) -> tuple[Tile, ...]:
        """Tiles in z, y, x order; built on first use, so counting and
        estimating a huge request stay cheap."""
        tiles = []
        for index in product(*(range(len(axis)) for axis in self.starts)):
            source, target, margins = [], [], []
            for axis, position in enumerate(index):
                count = len(self.starts[axis])
                start = self.starts[axis][position]
                seams = self.seams[axis]
                source.append(slice(start, start + self.tile_edge[axis]))
                target.append(
                    slice(
                        0 if position == 0 else seams[position - 1],
                        self.shape[axis] if position + 1 == count else seams[position],
                    )
                )
                overlap = self.overlap
                margins.append(
                    (
                        overlap if position > 0 else 0,
                        overlap if position + 1 < count else 0,
                    )
                )
            tiles.append(Tile(tuple(source), tuple(target), tuple(margins)))
        return tuple(tiles)

    @property
    def grid(self) -> tuple[int, int, int]:
        return tuple(len(axis) for axis in self.starts)

    def output_seams(self) -> tuple[tuple[int, ...], ...]:
        """Tile seams in output coordinates (inside the output only)."""
        return tuple(
            tuple(
                seam - self.margin
                for seam in axis
                if self.margin < seam < size + self.margin
            )
            for axis, size in zip(self.seams, self.output_shape, strict=True)
        )


def sr_margin(scale: int) -> int:
    """Default SR margin: 8 HR voxels, rounded up to whole LR cells."""
    return math.lcm(8, scale)


def parse_shape(value: int | Sequence[int], name: str = "shape") -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        shape = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        shape = tuple(value)
    else:
        raise TypeError(f"{name} must be an integer or three integers.")
    if len(shape) != 3 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in shape
    ):
        raise ValueError(f"{name} must contain exactly three positive integers.")
    return shape


def check_non_negative(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def make_plan(
    output_shape: Sequence[int],
    patch: int | Sequence[int],
    margin: int,
    overlap: int,
) -> TilePlan:
    output_shape = parse_shape(output_shape, "output shape")
    patch = parse_shape(patch, "patch")
    check_non_negative(margin, "margin")
    check_non_negative(overlap, "overlap")
    shape = tuple(size + 2 * margin for size in output_shape)
    edges = tuple(min(n + 2 * margin, g) for n, g in zip(patch, shape, strict=True))
    for edge, length in zip(edges, shape, strict=True):
        if length > edge and not 2 * overlap < edge:
            raise ValueError("overlap must satisfy 0 <= 2 * overlap < tile edge.")
    starts = tuple(
        _axis_starts(length, edge, edge - 2 * overlap)
        for length, edge in zip(shape, edges, strict=True)
    )
    seams = tuple(
        tuple((left + edge + right) // 2 for left, right in pairwise(axis))
        for axis, edge in zip(starts, edges, strict=True)
    )
    return TilePlan(
        output_shape=output_shape,
        margin=margin,
        overlap=overlap,
        tile_edge=edges,
        shape=shape,
        starts=starts,
        seams=seams,
    )


def _axis_starts(size: int, edge: int, stride: int) -> tuple[int, ...]:
    if size == edge:
        return (0,)
    count = math.ceil((size - edge) / stride) + 1
    starts = [index * stride for index in range(count - 1)]
    starts.append(size - edge)
    return tuple(starts)


def axis_window(
    length: int, overlap: int, left: int, right: int, device=None
) -> torch.Tensor:
    """1 inside a tile, a sin^2 ramp over `overlap` voxels on each seam side."""
    axis = torch.ones(length, device=device, dtype=torch.float32)
    if not overlap:
        return axis
    positions = torch.arange(overlap, device=device, dtype=torch.float32)
    ramp = torch.sin(positions * (math.pi / (2 * overlap))).square()
    if left:
        axis[:left] = ramp[-left:]
    if right:
        axis[-right:] = ramp.flip(0)[:right]
    return axis


def tile_window(tile: Tile, overlap: int, device=None) -> torch.Tensor:
    axes = [
        axis_window(length, overlap, left, right, device)
        for length, (left, right) in zip(tile.shape, tile.margins, strict=True)
    ]
    return (
        axes[0].view(1, 1, -1, 1, 1)
        * axes[1].view(1, 1, 1, -1, 1)
        * axes[2].view(1, 1, 1, 1, -1)
    )
