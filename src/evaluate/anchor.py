from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundaryQuality:
    anchor_change: float | None
    ordinary_change: float | None
    change_ratio: float | None


@dataclass(frozen=True)
class SliceSmoothness:
    p95_ratio: float | None
    max_ratio: float | None
    reversal_rate: float | None
    baseline_reversal_rate: float | None
    reversal_ratio: float | None
    peak_anchor_distance: int | None


def measure_boundaries(
    vol: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
) -> BoundaryQuality:
    if not indices:
        return BoundaryQuality(None, None, None)
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
        return BoundaryQuality(None, None, None)

    boundary_change = float(
        (slices[boundary_indices] != slices[[pair + 1 for pair in boundary_indices]])
        .to(torch.float32)
        .mean()
    )
    ordinary_change = float(
        (slices[ordinary_indices] != slices[[pair + 1 for pair in ordinary_indices]])
        .to(torch.float32)
        .mean()
    )
    ratio = None if ordinary_change == 0.0 else boundary_change / ordinary_change
    return BoundaryQuality(boundary_change, ordinary_change, ratio)


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
    positions = torch.arange(anchored_slices.shape[0], device=anchored_slices.device)
    anchors = torch.as_tensor(indices, dtype=torch.long, device=positions.device)
    distances = (positions[:, None] - anchors[None, :]).abs().amin(dim=1)
    return tuple(
        None
        if not bool((distances == distance).any())
        else float(changes[distances == distance].mean())
        for distance in range(max_distance + 1)
    )


def measure_slice_smoothness(
    vol: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    baseline: torch.Tensor | None = None,
) -> SliceSmoothness:
    acceleration = _slice_acceleration(vol, axis)
    if acceleration.numel() == 0:
        return SliceSmoothness(None, None, None, None, None, None)

    p95 = float(torch.quantile(acceleration, 0.95))
    maximum, peak = acceleration.max(dim=0)
    peak_index = int(peak) + 1
    peak_distance = (
        None if not indices else min(abs(peak_index - index) for index in indices)
    )
    reversal_rate = _slice_reversal_rate(vol, axis)
    baseline_p95 = None
    baseline_max = None
    baseline_reversal_rate = None
    p95_ratio = None
    max_ratio = None
    reversal_ratio = None
    if baseline is not None:
        if baseline.shape != vol.shape:
            raise ValueError("baseline must have the same shape as volume.")
        baseline_acceleration = _slice_acceleration(baseline, axis)
        baseline_p95 = float(torch.quantile(baseline_acceleration, 0.95))
        baseline_max = float(baseline_acceleration.max())
        baseline_reversal_rate = _slice_reversal_rate(baseline, axis)
        if baseline_p95 > 0.0:
            p95_ratio = p95 / baseline_p95
        if baseline_max > 0.0:
            max_ratio = float(maximum) / baseline_max
        if baseline_reversal_rate > 0.0:
            reversal_ratio = reversal_rate / baseline_reversal_rate
    return SliceSmoothness(
        p95_ratio,
        max_ratio,
        reversal_rate,
        baseline_reversal_rate,
        reversal_ratio,
        peak_distance,
    )


def _get_slices(vol: torch.Tensor, axis: int) -> torch.Tensor:
    return vol.movedim(axis, 0)


def _slice_acceleration(vol: torch.Tensor, axis: int) -> torch.Tensor:
    slices = _get_slices(vol, axis)
    change = (slices[1:] != slices[:-1]).to(torch.float32).mean((1, 2))
    if change.numel() < 2:
        return change.new_empty(0)
    return (change[1:] - change[:-1]).abs()


def _slice_reversal_rate(vol: torch.Tensor, axis: int) -> float:
    slices = _get_slices(vol, axis)
    if slices.shape[0] < 3:
        return 0.0
    reversals = (slices[:-2] == slices[2:]) & (slices[1:-1] != slices[:-2])
    return float(reversals.to(torch.float32).mean())
