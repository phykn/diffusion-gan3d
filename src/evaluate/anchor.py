from dataclasses import dataclass

import torch

from .connect import (
    continuation_delta,
    phase_change_rate,
    transition_counts,
    transition_tv,
)


@dataclass(frozen=True)
class BoundaryQuality:
    anchor_change: float | None
    ordinary_change: float | None
    change_ratio: float | None
    transition_tv: float | None
    continuation_delta: float | None


def measure_boundaries(
    vol: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    num_phases: int,
) -> BoundaryQuality:
    slices = _get_slices(vol, axis)
    pair_count = slices.shape[0] - 1
    boundary_indices = sorted(
        {
            pair
            for index in indices
            for pair in (index - 1, index)
            if 0 <= pair < pair_count
        }
    )
    boundary_set = set(boundary_indices)
    ordinary_indices = [pair for pair in range(pair_count) if pair not in boundary_set]
    if not boundary_indices or not ordinary_indices:
        return BoundaryQuality(None, None, None, None, None)

    boundary_counts = transition_counts(
        slices[boundary_indices],
        slices[[index + 1 for index in boundary_indices]],
        num_phases,
    )
    ordinary_counts = transition_counts(
        slices[ordinary_indices],
        slices[[index + 1 for index in ordinary_indices]],
        num_phases,
    )
    boundary_change = phase_change_rate(boundary_counts)
    ordinary_change = phase_change_rate(ordinary_counts)
    ratio = None if ordinary_change == 0.0 else boundary_change / ordinary_change
    return BoundaryQuality(
        boundary_change,
        ordinary_change,
        ratio,
        transition_tv(boundary_counts, ordinary_counts),
        continuation_delta(boundary_counts, ordinary_counts),
    )


def measure_distance_changes(
    vol: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    max_distance: int,
) -> tuple[float | None, ...]:
    if not indices:
        return ()
    slices = _get_slices(vol, axis)
    buckets: list[list[int]] = [[] for _ in range(max_distance + 1)]
    for pair in range(slices.shape[0] - 1):
        distance = min(
            min(abs(pair - index), abs(pair + 1 - index)) for index in indices
        )
        if distance <= max_distance:
            buckets[distance].append(pair)
    profile = []
    for pairs in buckets:
        if not pairs:
            profile.append(None)
            continue
        left = slices[pairs]
        right = slices[[pair + 1 for pair in pairs]]
        profile.append(float((left != right).to(torch.float32).mean()))
    return tuple(profile)


def measure_distance_divergence(
    anchored: torch.Tensor,
    baseline: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    max_distance: int,
) -> tuple[float | None, ...]:
    if not indices:
        return ()
    anchored_slices = _get_slices(anchored, axis)
    baseline_slices = _get_slices(baseline, axis)
    changes = (anchored_slices != baseline_slices).to(torch.float32).mean((1, 2))
    positions = torch.arange(anchored_slices.shape[0], dtype=torch.long)
    anchors = torch.tensor(indices, dtype=torch.long)
    distances = (positions[:, None] - anchors[None, :]).abs().amin(dim=1)
    profile = []
    for distance in range(max_distance + 1):
        selected = changes[distances == distance]
        profile.append(None if selected.numel() == 0 else float(selected.mean()))
    return tuple(profile)


def _get_slices(vol: torch.Tensor, axis: int) -> torch.Tensor:
    if vol.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return vol.movedim(axis, 0)
