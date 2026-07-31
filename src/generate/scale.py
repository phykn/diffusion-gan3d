from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product

import torch
from tqdm import tqdm

from ..anchor import PlaneAnchor
from ..model import Denoiser3D
from ..train import TrainConfig
from .sample import Sampler


@dataclass(frozen=True)
class ScaleStats:
    shape: tuple[int, int, int]
    block_grid: tuple[int, int, int]
    block_size: int
    overlap: int
    block_count: int
    anchor_planes: int
    anchor_accuracy: float | None
    anchor_accuracy_by_count: tuple[tuple[int, float], ...]
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@torch.no_grad()
def generate_scaled(
    model: Denoiser3D,
    cfg: TrainConfig,
    *,
    device: torch.device,
    blocks: int | Sequence[int],
    overlap: int | None = None,
    mixed_precision: bool | None = None,
    progress: bool = True,
) -> tuple[torch.Tensor, ScaleStats]:
    grid = _parse_grid(blocks)
    block_size = cfg.data.patch_size
    overlap = block_size // 2 if overlap is None else overlap
    if (
        not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or not 1 <= overlap < block_size
    ):
        raise ValueError("overlap must be an integer between 1 and block_size - 1.")
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean.")

    stride = block_size - overlap
    starts = tuple(tuple(index * stride for index in range(count)) for count in grid)
    shape = tuple(block_size + (count - 1) * stride for count in grid)
    indices = tuple(product(*(range(count) for count in grid)))
    if len(indices) > 1 and not cfg.anchor.enabled:
        raise ValueError("scaled generation requires anchor-aware weights.")

    sampler = Sampler(
        model,
        cfg,
        device=device,
        mixed_precision=mixed_precision,
    )
    tiles: dict[tuple[int, int, int], torch.Tensor] = {}
    correct: dict[int, int] = {}
    compared: dict[int, int] = {}
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
            blocks=tiles,
            block_size=block_size,
        )
        labels = sampler.generate(
            size=block_size,
            anchors=anchors,
        )
        if anchors:
            matched = sum(
                int((labels.select(anchor.axis, anchor.index) == anchor.labels).sum())
                for anchor in anchors
            )
            total = len(anchors) * block_size * block_size
            count = len(anchors)
            correct[count] = correct.get(count, 0) + matched
            compared[count] = compared.get(count, 0) + total
            anchor_planes += count
            # Exact cached intersections prevent later multi-plane conflicts.
            _project_anchors(labels, anchors)
        tiles[index] = labels

    labels = _assemble(
        tiles,
        starts=starts,
        shape=shape,
        block_size=block_size,
    )
    total_correct = sum(correct.values())
    total_compared = sum(compared.values())
    stats = ScaleStats(
        shape=shape,
        block_grid=grid,
        block_size=block_size,
        overlap=overlap,
        block_count=len(tiles),
        anchor_planes=anchor_planes,
        anchor_accuracy=None if total_compared == 0 else total_correct / total_compared,
        anchor_accuracy_by_count=tuple(
            (count, correct[count] / compared[count]) for count in sorted(correct)
        ),
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
    blocks: dict[tuple[int, int, int], torch.Tensor],
    block_size: int,
) -> tuple[PlaneAnchor, ...]:
    origin = tuple(starts[axis][index[axis]] for axis in range(3))
    anchors = []
    for axis in range(3):
        if index[axis] == 0:
            continue
        previous = list(index)
        previous[axis] -= 1
        previous_index = tuple(previous)
        previous_origin = starts[axis][previous_index[axis]]
        current_origin = origin[axis]
        actual_overlap = block_size - (current_origin - previous_origin)
        seam = current_origin + actual_overlap // 2
        source_index = seam - previous_origin
        target_index = seam - current_origin
        source = blocks[previous_index].select(axis, source_index)
        anchors.append(
            PlaneAnchor(
                labels=source.to(dtype=torch.long).clone(),
                axis=axis,
                index=target_index,
            )
        )
    return tuple(anchors)


def _project_anchors(
    labels: torch.Tensor,
    anchors: Sequence[PlaneAnchor],
) -> None:
    for anchor in anchors:
        labels.select(anchor.axis, anchor.index).copy_(
            anchor.labels.to(dtype=labels.dtype)
        )


def _split_axis(
    starts: tuple[int, ...],
    block_size: int,
    size: int,
) -> tuple[tuple[int, int], ...]:
    seams = tuple((left + right + block_size) // 2 for left, right in pairwise(starts))
    boundaries = (0, *seams, size)
    return tuple(pairwise(boundaries))


def _assemble(
    blocks: dict[tuple[int, int, int], torch.Tensor],
    *,
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    shape: tuple[int, int, int],
    block_size: int,
) -> torch.Tensor:
    cells = tuple(
        _split_axis(values, block_size, length)
        for values, length in zip(starts, shape, strict=True)
    )
    output = torch.empty(shape, dtype=torch.uint8)
    for index, block in blocks.items():
        global_slices = []
        local_slices = []
        for axis in range(3):
            start, stop = cells[axis][index[axis]]
            global_slices.append(slice(start, stop))
            local_start = start - starts[axis][index[axis]]
            local_slices.append(slice(local_start, local_start + stop - start))
        output[tuple(global_slices)] = block[tuple(local_slices)]
    return output
