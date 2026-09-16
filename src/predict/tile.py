import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product

import torch


@dataclass(frozen=True)
class TilePlan:
    shape: tuple[int, int, int]
    tile_size: int
    overlap: int
    stride: int
    grid: tuple[int, int, int]
    tile_count: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    generation_shape: tuple[int, int, int] | None = None
    margin: int = 0

    @property
    def base_shell(self) -> int:
        return min(self.overlap // 2, (self.stride - 1) // 2)


@dataclass(frozen=True)
class Tile:
    source: tuple[slice, slice, slice]
    target: tuple[slice, slice, slice]
    margins: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class Fusion:
    axis_windows: dict[tuple[int, int, int], torch.Tensor]
    weight_sum: torch.Tensor
    pred_sum: torch.Tensor

    def regions(self, region: tuple[slice, slice, slice]):
        """Map a global z interval into the bounded, circular accumulation slab."""
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


class VolumeState:
    def __init__(
        self,
        num_phases: int,
        shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        self.values = torch.empty(
            (1, num_phases, *shape),
            device=device,
            dtype=torch.float16,
        )

    def read(
        self,
        region: tuple[slice, slice, slice],
    ) -> torch.Tensor:
        return self.values[(slice(None), slice(None), *region)]

    def write(
        self,
        region: tuple[slice, slice, slice],
        values: torch.Tensor,
    ) -> None:
        self.values[(slice(None), slice(None), *region)].copy_(
            values.to(device=self.values.device, dtype=torch.float16)
        )


class TileBuffer:
    def __init__(
        self,
        num_phases: int,
        tile_size: int,
        enabled: bool,
    ) -> None:
        self.upload: torch.Tensor | None = None
        self.download: torch.Tensor | None = None
        self.workspace: torch.Tensor | None = None
        self.capacity = num_phases * tile_size**3
        if enabled:
            try:
                self.upload = torch.empty(
                    self.capacity,
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self.download = torch.empty(
                    self.capacity,
                    dtype=torch.float32,
                    pin_memory=True,
                )
            except RuntimeError:
                self.upload = None
                self.download = None

    def read(
        self,
        state: VolumeState,
        region: tuple[slice, slice, slice],
        device: torch.device,
    ) -> torch.Tensor:
        source = state.read(region)
        shape = source.shape
        numel = math.prod(shape)
        if state.values.device != device and self.upload is not None:
            values = self.upload[:numel].view(shape)
        else:
            if (
                self.workspace is None
                or self.workspace.device != state.values.device
                or self.workspace.numel() < numel
            ):
                self.workspace = torch.empty(
                    max(self.capacity, numel),
                    device=state.values.device,
                    dtype=torch.float32,
                )
            values = self.workspace[:numel].view(shape)

        values.copy_(source)
        if values.device == device:
            return values
        return values.to(
            device=device,
            dtype=torch.float32,
            non_blocking=self.upload is not None,
        )

    def stage(self, values: torch.Tensor, device: torch.device) -> torch.Tensor:
        if values.device == device:
            return values
        if (
            device.type == "cpu"
            and self.download is not None
            and values.numel() <= self.download.numel()
        ):
            downloaded = self.download[: values.numel()].view(values.shape)
            downloaded.copy_(values)
            return downloaded
        return values.to(device)


def output_plan(
    plan: TilePlan,
    output_shape: tuple[int, int, int],
) -> TilePlan:
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
    starts = tuple(
        axis_starts(size, plan.tile_size, plan.stride) for size in plan.shape
    )
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
            source_start = starts[axis][tile_idx]
            source_stop = source_start + plan.tile_size
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


def write_output(
    labels: torch.Tensor,
    target: tuple[slice, slice, slice],
    clean: torch.Tensor,
) -> None:
    values = clean.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
    labels[target].copy_(values)
