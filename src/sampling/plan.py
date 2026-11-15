import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise, product

import torch


@dataclass(frozen=True)
class Tile:
    # The denoiser reads `source`; the tile alone writes the result of `target`.
    source: tuple
    target: tuple
    blend_widths: tuple

    @property
    def shape(self):
        return tuple(region.stop - region.start for region in self.source)


@dataclass(frozen=True)
class TilePlan:
    # All in (z, y, x). The region is the output plus `margin` on the y and x
    # faces and `z_margins` (top, bottom) on the z faces.
    output_shape: tuple
    margin: int
    z_margins: tuple
    overlap: int
    tile_shape: tuple
    region_shape: tuple
    tile_starts: tuple
    seams: tuple

    @property
    def offsets(self):
        """Where the output starts in the region, per axis."""
        return (self.z_margins[0], self.margin, self.margin)

    @property
    def tiles_per_axis(self):
        return tuple(len(starts) for starts in self.tile_starts)

    @property
    def tile_count(self):
        return math.prod(self.tiles_per_axis)

    # Built on first use, so counting the tiles of a huge request stays cheap.
    @cached_property
    def tiles(self):
        tiles = []
        # z varies slowest: the sampler finishes one z layer of tiles at a time.
        for positions in product(*(range(count) for count in self.tiles_per_axis)):
            source, target, blend_widths = [], [], []
            for axis, position in enumerate(positions):
                is_first = position == 0
                is_last = position + 1 == self.tiles_per_axis[axis]
                start = self.tile_starts[axis][position]
                seams = self.seams[axis]
                source.append(slice(start, start + self.tile_shape[axis]))
                target.append(
                    slice(
                        0 if is_first else seams[position - 1],
                        self.region_shape[axis] if is_last else seams[position],
                    )
                )
                blend_widths.append(
                    (0 if is_first else self.overlap, 0 if is_last else self.overlap)
                )
            tiles.append(Tile(tuple(source), tuple(target), tuple(blend_widths)))
        return tuple(tiles)

    @property
    def output_seams(self):
        return tuple(
            tuple(seam - offset for seam in seams if offset < seam < size + offset)
            for seams, size, offset in zip(
                self.seams, self.output_shape, self.offsets, strict=True
            )
        )


def make_plan(output_shape, patch, margin, overlap, z_margins=None):
    """z_margins (top, bottom) default to margin; a height-conditioned volume
    keeps them inside the source height (src/sampling/conditions.py)."""
    output_shape = parse_shape(output_shape, "output shape")
    patch = parse_shape(patch, "patch")
    check_non_negative(margin, "margin")
    check_non_negative(overlap, "overlap")
    z_margins = (margin, margin) if z_margins is None else tuple(z_margins)
    for value in z_margins:
        check_non_negative(value, "z margin")
    widths = (sum(z_margins), 2 * margin, 2 * margin)
    region_shape = tuple(
        size + width for size, width in zip(output_shape, widths, strict=True)
    )
    # Tiles keep one size on every axis; only the region loses cut z margins.
    tile_shape = tuple(
        min(size + 2 * margin, region)
        for size, region in zip(patch, region_shape, strict=True)
    )
    for edge, region in zip(tile_shape, region_shape, strict=True):
        if region > edge and not 2 * overlap < edge:
            raise ValueError("overlap must satisfy 0 <= 2 * overlap < tile edge.")
    tile_starts = tuple(
        _list_starts(region, edge, edge - 2 * overlap)
        for region, edge in zip(region_shape, tile_shape, strict=True)
    )
    seams = tuple(
        tuple((left + edge + right) // 2 for left, right in pairwise(starts))
        for starts, edge in zip(tile_starts, tile_shape, strict=True)
    )
    return TilePlan(
        output_shape=output_shape,
        margin=margin,
        z_margins=z_margins,
        overlap=overlap,
        tile_shape=tile_shape,
        region_shape=region_shape,
        tile_starts=tile_starts,
        seams=seams,
    )


def parse_shape(value, name):
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


def check_non_negative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def choose_sr_margin(scale):
    # Eight voxels, rounded up to whole LR cells.
    return math.lcm(8, scale)


def build_tile_window(tile, overlap, device):
    z, y, x = (
        _build_axis_window(length, overlap, before, after, device)
        for length, (before, after) in zip(tile.shape, tile.blend_widths, strict=True)
    )
    return z.view(1, 1, -1, 1, 1) * y.view(1, 1, 1, -1, 1) * x.view(1, 1, 1, 1, -1)


def _list_starts(region, edge, stride):
    if region == edge:
        return (0,)
    last = region - edge
    return (*range(0, last, stride), last)


def _build_axis_window(length, overlap, before, after, device):
    # 1 inside the tile, a sin^2 ramp over the overlap toward each neighbour.
    window = torch.ones(length, device=device, dtype=torch.float32)
    if not overlap:
        return window
    positions = torch.arange(overlap, device=device, dtype=torch.float32)
    ramp = torch.sin(positions * (math.pi / (2 * overlap))).square()
    # A blend width is 0 or the whole overlap.
    if before:
        window[:before] = ramp
    if after:
        window[-after:] = ramp.flip(0)
    return window
