import math
from dataclasses import dataclass

import torch

from src.predict.tiling.layout import Tile, TilePlan
from src.predict.tiling.state import TileBuffer


@dataclass(frozen=True)
class Fusion:
    axis_windows: dict[tuple[int, int, int], torch.Tensor]
    weight_sum: torch.Tensor
    pred_sum: torch.Tensor

    def regions(self, region: tuple[slice, slice, slice]):
        depth = self.pred_sum.shape[2]
        start, stop = region[0].start, region[0].stop
        while start < stop:
            offset = start % depth
            end = min(stop, start + depth - offset)
            yield (
                (slice(start, end), *region[1:]),
                (slice(offset, offset + end - start), *region[1:]),
            )
            start = end


def make_fusion(
    plan: TilePlan,
    tiles: tuple[Tile, ...],
    num_phases: int,
    device: torch.device,
    tile_device: torch.device,
) -> Fusion:
    # Tiles are traversed in z/y/x order. Once the next z layer starts, no
    # future prediction can affect earlier voxels, so a single tile-depth
    # circular slab suffices for both weighted sums.
    axis_windows: dict[tuple[int, int, int], torch.Tensor] = {}
    shape = (min(plan.tile_size, plan.shape[0]), *plan.shape[1:])
    weight_sum = torch.zeros(
        (1, 1, *shape),
        device=device,
        dtype=torch.float32,
    )
    pred_sum = torch.zeros(
        (1, num_phases, *shape),
        device=device,
        dtype=torch.float32,
    )
    return Fusion(
        axis_windows=axis_windows,
        weight_sum=weight_sum,
        pred_sum=pred_sum,
    )


def get_axis_windows(
    tile: Tile,
    overlap: int,
    device: torch.device,
    cache: dict[tuple[int, int, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    windows = []
    for region, (left_margin, right_margin) in zip(
        tile.source,
        tile.margins,
        strict=True,
    ):
        length = region.stop - region.start
        key = (length, left_margin, right_margin)
        if key not in cache:
            cache[key] = make_axis_window(
                length,
                overlap,
                left_margin,
                right_margin,
                device,
            )
        windows.append(cache[key])
    return tuple(windows)


def make_axis_window(
    length: int,
    overlap: int,
    left_margin: int,
    right_margin: int,
    device: torch.device,
) -> torch.Tensor:
    axis = torch.ones(length, device=device, dtype=torch.float32)
    if not overlap:
        return axis
    positions = torch.arange(overlap, device=device, dtype=torch.float32)
    ramp = torch.sin(positions * (math.pi / (2 * overlap))).square()
    if left_margin:
        axis[:left_margin] = ramp[-left_margin:]
    if right_margin:
        axis[-right_margin:] = ramp.flip(0)[:right_margin]
    return axis


def add_prediction(
    fusion: Fusion,
    tile: Tile,
    pred: torch.Tensor,
    tile_buffer: TileBuffer,
    overlap: int,
) -> None:
    weighted = pred.float().clone()
    axes = get_axis_windows(
        tile,
        overlap,
        pred.device,
        fusion.axis_windows,
    )
    for spatial_axis, axis in enumerate(axes, 2):
        shape = [1, 1, 1, 1, 1]
        shape[spatial_axis] = axis.numel()
        weighted.mul_(axis.view(shape))
    staged = tile_buffer.stage(weighted, fusion.pred_sum.device)
    window = (
        axes[0].view(1, 1, -1, 1, 1)
        * axes[1].view(1, 1, 1, -1, 1)
        * axes[2].view(1, 1, 1, 1, -1)
    ).to(fusion.weight_sum.device)
    for global_region, local_region in fusion.regions(tile.source):
        start = global_region[0].start - tile.source[0].start
        stop = global_region[0].stop - tile.source[0].start
        target = (slice(None), slice(None), *local_region)
        fusion.pred_sum[target].add_(staged[:, :, start:stop])
        fusion.weight_sum[target].add_(window[:, :, start:stop])
