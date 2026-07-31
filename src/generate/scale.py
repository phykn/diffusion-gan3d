import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product

import torch
from tqdm import tqdm

from .sample import Sampler


@dataclass(frozen=True)
class ScaleStats:
    shape: tuple[int, int, int]
    block_grid: tuple[int, int, int]
    block_size: int
    overlap: int
    block_count: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class _Tile:
    region: tuple[slice, slice, slice]
    roles: tuple[int, int, int]


_SINGLE = 0
_START = 1
_MIDDLE = 2
_END = 3


@torch.no_grad()
def generate_scaled(
    sampler: Sampler,
    *,
    blocks: int | Sequence[int],
    fraction: Sequence[float] | None = None,
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
    fraction_tensor = sampler.prepare_fraction(fraction)

    stride = block_size - overlap
    starts = tuple(tuple(index * stride for index in range(count)) for count in grid)
    shape = tuple(block_size + (count - 1) * stride for count in grid)
    tiles = _make_tiles(
        starts,
        grid=grid,
        block_size=block_size,
    )
    axis_weights = _make_axis_weights(
        block_size,
        overlap,
        device=sampler.device,
    )

    current = torch.randn(
        1,
        sampler.num_phases,
        *shape,
        device=sampler.device,
        dtype=torch.float32,
    )
    bar = tqdm(
        reversed(range(sampler.diffusion.timesteps)),
        total=sampler.diffusion.timesteps,
        desc="Scale up",
        disable=not progress,
    )
    times = torch.empty(1, device=sampler.device, dtype=torch.long)
    clean_sum = torch.empty_like(current)
    weighted = torch.empty(
        1,
        sampler.num_phases,
        block_size,
        block_size,
        block_size,
        device=sampler.device,
        dtype=torch.float32,
    )
    for transition in bar:
        times.fill_(transition)
        latent = torch.randn(
            1,
            sampler.latent_channels,
            device=sampler.device,
            dtype=current.dtype,
        )
        clean_sum.zero_()
        for tile in tiles:
            _accumulate_prediction(
                sampler,
                current=current,
                clean_sum=clean_sum,
                weighted=weighted,
                tile=tile,
                axis_weights=axis_weights,
                times=times,
                latent=latent,
                fraction=fraction_tensor,
            )

        current = sampler.diffusion.sample_posterior(
            current,
            clean_sum,
            transition,
        )

    probabilities = (current.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)
    probabilities.div_(
        probabilities.sum(dim=1, keepdim=True).clamp_min_(
            torch.finfo(probabilities.dtype).eps
        )
    )
    labels = probabilities.argmax(dim=1).squeeze(0).cpu().to(torch.uint8)
    stats = ScaleStats(
        shape=shape,
        block_grid=grid,
        block_size=block_size,
        overlap=overlap,
        block_count=len(tiles),
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


def _make_tiles(
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    *,
    grid: tuple[int, int, int],
    block_size: int,
) -> tuple[_Tile, ...]:
    tiles = []
    for index in product(*(range(count) for count in grid)):
        region = tuple(
            slice(
                starts[axis][index[axis]],
                starts[axis][index[axis]] + block_size,
            )
            for axis in range(3)
        )
        tiles.append(
            _Tile(
                region=region,
                roles=tuple(_get_role(index[axis], grid[axis]) for axis in range(3)),
            )
        )
    return tuple(tiles)


def _make_axis_weights(
    block_size: int,
    overlap: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(overlap, device=device, dtype=torch.float32)
    rise = torch.sin((positions + 0.5) * math.pi / (2.0 * overlap)).square()
    fall = 1.0 - rise
    single = torch.ones(block_size, device=device, dtype=torch.float32)
    start = single.clone()
    start[-overlap:] = fall
    middle = start.clone()
    middle[:overlap] = rise
    end = single.clone()
    end[:overlap] = rise
    return single, start, middle, end


def _get_role(index: int, count: int) -> int:
    if count == 1:
        return _SINGLE
    if index == 0:
        return _START
    if index + 1 == count:
        return _END
    return _MIDDLE


def _accumulate_prediction(
    sampler: Sampler,
    *,
    current: torch.Tensor,
    clean_sum: torch.Tensor,
    weighted: torch.Tensor,
    tile: _Tile,
    axis_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    times: torch.Tensor,
    latent: torch.Tensor,
    fraction: torch.Tensor | None,
) -> None:
    key = (slice(None), slice(None), *tile.region)
    current_tile = current[key]
    with torch.autocast(
        device_type=sampler.device.type,
        dtype=torch.float16,
        enabled=sampler.use_amp,
    ):
        if fraction is None:
            clean_tile = sampler.model(current_tile, times, latent)
        else:
            clean_tile = sampler.model(
                current_tile,
                times,
                latent,
                fraction=fraction,
            )

    weighted.copy_(clean_tile)
    weighted.mul_(axis_weights[tile.roles[0]].view(1, 1, -1, 1, 1))
    weighted.mul_(axis_weights[tile.roles[1]].view(1, 1, 1, -1, 1))
    weighted.mul_(axis_weights[tile.roles[2]].view(1, 1, 1, 1, -1))
    clean_sum[key].add_(weighted)


def _split_axis(
    starts: tuple[int, ...],
    block_size: int,
    size: int,
) -> tuple[tuple[int, int], ...]:
    seams = tuple((left + right + block_size) // 2 for left, right in pairwise(starts))
    boundaries = (0, *seams, size)
    return tuple(pairwise(boundaries))
