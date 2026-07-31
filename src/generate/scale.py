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
    weight: torch.Tensor


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
    tiles = _make_tiles(
        starts,
        grid=grid,
        block_size=block_size,
        overlap=overlap,
        device=sampler.device,
    )
    weight_sum = torch.zeros(
        1,
        1,
        *shape,
        device=sampler.device,
        dtype=torch.float32,
    )
    for tile in tiles:
        key = (slice(None), slice(None), *tile.region)
        weight_sum[key].add_(tile.weight)
    if bool((weight_sum <= 0.0).any().item()):
        raise RuntimeError("tile weights must cover the complete scaled volume.")
    inverse_weight = weight_sum.reciprocal_()

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
    for transition in bar:
        times.fill_(transition)
        latent = torch.randn(
            1,
            sampler.latent_channels,
            device=sampler.device,
            dtype=current.dtype,
        )
        clean_sum = torch.zeros_like(current)
        for tile in tiles:
            key = (slice(None), slice(None), *tile.region)
            current_tile = current[key]
            with torch.autocast(
                device_type=sampler.device.type,
                dtype=torch.float16,
                enabled=sampler.use_amp,
            ):
                clean_tile = sampler.model(current_tile, times, latent)
            if not isinstance(clean_tile, torch.Tensor):
                raise TypeError("model must return a torch.Tensor.")
            if clean_tile.shape != current_tile.shape:
                raise ValueError("model output must match the input tile shape.")
            if clean_tile.device != current_tile.device:
                raise ValueError("model output must use the input tile device.")
            if not clean_tile.is_floating_point():
                raise ValueError("model output must be floating point.")
            clean_sum[key].add_(clean_tile.float() * tile.weight)

        clean = clean_sum * inverse_weight
        posterior_noise = None if transition == 0 else torch.randn_like(current)
        current = sampler.diffusion.sample_posterior(
            current,
            clean,
            transition,
            noise=posterior_noise,
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
    overlap: int,
    device: torch.device,
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
        weight = _make_weight(
            index,
            grid=grid,
            block_size=block_size,
            overlap=overlap,
            dtype=torch.float32,
        ).to(device=device)
        tiles.append(
            _Tile(
                region=region,
                weight=weight.unsqueeze(0).unsqueeze(0),
            )
        )
    return tuple(tiles)


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
