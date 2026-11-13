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


def measure_boundaries(vol, indices, axis):
    if not indices:
        return BoundaryQuality(None, None, None)
    slices = vol.movedim(axis, 0)
    pair_count = slices.shape[0] - 1
    boundaries = {
        pair
        for index in indices
        for pair in (index - 1, index)
        if 0 <= pair < pair_count
    }
    ordinary = sorted(set(range(pair_count)) - boundaries)
    if not boundaries or not ordinary:
        return BoundaryQuality(None, None, None)

    boundary_change = _change(slices, sorted(boundaries))
    ordinary_change = _change(slices, ordinary)
    ratio = None if ordinary_change == 0.0 else boundary_change / ordinary_change
    return BoundaryQuality(boundary_change, ordinary_change, ratio)


def measure_distance_divergence(anchored, baseline, indices, axis):
    if not indices:
        return ()
    anchored_slices = anchored.movedim(axis, 0)
    baseline_slices = baseline.movedim(axis, 0)
    changes = (anchored_slices != baseline_slices).to(torch.float32).mean((1, 2))
    positions = torch.arange(anchored_slices.shape[0], device=anchored_slices.device)
    anchors = torch.as_tensor(indices, dtype=torch.long, device=positions.device)
    distances = (positions[:, None] - anchors[None, :]).abs().amin(dim=1)
    result = [None] * anchored_slices.shape[0]
    for distance in distances.unique().tolist():
        if distance < len(result):
            result[distance] = float(changes[distances == distance].mean())
    return tuple(result)


def measure_slice_smoothness(vol, indices, axis, baseline=None):
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


def _change(slices, pairs):
    following = [pair + 1 for pair in pairs]
    return float((slices[pairs] != slices[following]).to(torch.float32).mean())


def _slice_acceleration(vol, axis):
    slices = vol.movedim(axis, 0)
    change = (slices[1:] != slices[:-1]).to(torch.float32).mean((1, 2))
    return change.diff().abs()


def _slice_reversal_rate(vol, axis):
    slices = vol.movedim(axis, 0)
    if slices.shape[0] < 3:
        return 0.0
    reversals = (slices[:-2] == slices[2:]) & (slices[1:-1] != slices[:-2])
    return float(reversals.to(torch.float32).mean())
