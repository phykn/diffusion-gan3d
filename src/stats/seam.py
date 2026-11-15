from dataclasses import dataclass

import numpy as np
import torch

from src.device import wide_dtype
from src.stats.labels import phase_labels


@dataclass(frozen=True)
class SeamQuality:
    transition_tv: tuple[float | None, float | None, float | None]
    continuation_delta: tuple[float | None, float | None, float | None]


def measure_seams(vol, seams, band_size, num_phases):
    tvs = []
    deltas = []
    width = max(1, min(band_size, 4))
    for axis, positions in enumerate(seams):
        if not positions:
            tvs.append(None)
            deltas.append(None)
            continue
        pair_count = vol.shape[axis] - 1
        band = {
            idx
            for pos in positions
            for idx in range(max(0, pos - width), min(pair_count, pos + width))
        }
        interior = sorted(set(range(pair_count)) - band)
        if not interior or not band:
            tvs.append(None)
            deltas.append(None)
            continue
        if len(interior) > 64:
            selected = np.linspace(0, len(interior) - 1, num=64, dtype=int)
            interior = [interior[idx] for idx in selected]
        inner_counts = torch.zeros(
            num_phases,
            num_phases,
            dtype=wide_dtype(vol.device),
            device=vol.device,
        )
        band_counts = {}
        sections = vol.movedim(axis, 0)
        stride = max(1, (max(sections.shape[-2:]) + 511) // 512)
        sections = sections[:, ::stride, ::stride]
        for idx in sorted(band | set(interior)):
            counts = transition_counts(sections[idx], sections[idx + 1], num_phases)
            if idx in band:
                band_counts[idx] = counts
            else:
                inner_counts.add_(counts)

        axis_tv = []
        axis_delta = []
        for seam_counts in band_counts.values():
            axis_tv.append(transition_tv(seam_counts, inner_counts))
            delta = continuation_delta(seam_counts, inner_counts)
            if delta is not None:
                axis_delta.append(delta)
        tvs.append(max(axis_tv))
        deltas.append(max(axis_delta) if axis_delta else None)
    return SeamQuality(
        transition_tv=tuple(tvs),
        continuation_delta=tuple(deltas),
    )


def transition_counts(previous, current, num_phases):
    previous = phase_labels(previous, num_phases)
    current = phase_labels(current, num_phases).to(previous.device)
    if previous.shape != current.shape:
        raise ValueError("transition labels must have the same shape.")
    pairs = previous.reshape(-1) * num_phases + current.reshape(-1)
    counts = torch.bincount(pairs, minlength=num_phases * num_phases)
    return counts.reshape(num_phases, num_phases).to(wide_dtype(counts.device))


def transition_tv(first, second):
    first, second = _transition_matrices(first, second)
    first = first / first.sum()
    second = second / second.sum()
    return float(0.5 * (first - second).abs().sum())


def continuation_delta(first, second):
    first, second = _transition_matrices(first, second)
    first_totals = first.sum(dim=1)
    second_totals = second.sum(dim=1)
    supported = (first_totals > 0) & (second_totals > 0)
    if not bool(supported.any()):
        return None
    first_values = first.diagonal()[supported] / first_totals[supported]
    second_values = second.diagonal()[supported] / second_totals[supported]
    return float((first_values - second_values).abs().max())


def _transition_matrices(first, second):
    first = torch.as_tensor(first)
    first = first.to(wide_dtype(first.device))
    second = torch.as_tensor(second, dtype=first.dtype, device=first.device)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or first.shape != second.shape
        or first.numel() == 0
    ):
        raise ValueError(
            "transition matrices must have the same non-empty square shape."
        )
    for counts in (first, second):
        if not bool((torch.isfinite(counts) & (counts >= 0)).all()) or not bool(
            counts.sum() > 0
        ):
            raise ValueError(
                "transition counts must be finite, non-negative and nonzero."
            )
    return first, second
