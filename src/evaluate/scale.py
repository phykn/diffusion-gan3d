from dataclasses import dataclass

import numpy as np
import torch

from .connect import (
    continuation_delta,
    transition_counts,
    transition_tv,
)


@dataclass(frozen=True)
class SeamQuality:
    transition_tv: tuple[float | None, float | None, float | None]
    continuation_delta: tuple[float | None, float | None, float | None]


def measure_seams(
    vol: torch.Tensor,
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    band_size: int,
    num_phases: int,
) -> SeamQuality:
    tvs = []
    deltas = []
    for axis, positions in enumerate(seams):
        if not positions:
            tvs.append(None)
            deltas.append(None)
            continue
        pair_count = vol.shape[axis] - 1
        width = max(1, min(band_size, 4))
        band_idx = sorted(
            {
                idx
                for pos in positions
                for idx in range(
                    max(0, pos - width),
                    min(pair_count, pos + width),
                )
            }
        )
        band_set = set(band_idx)
        inner_idx = [idx for idx in range(pair_count) if idx not in band_set]
        if len(inner_idx) > 64:
            selected = np.linspace(0, len(inner_idx) - 1, num=64, dtype=int)
            inner_idx = [inner_idx[idx] for idx in selected]
        selected_idx = sorted(band_set | set(inner_idx))
        inner_set = set(inner_idx)
        inner_counts = torch.zeros(
            num_phases,
            num_phases,
            dtype=torch.float64,
        )
        band_counts = {}
        for idx in selected_idx:
            prev = vol.select(axis, idx)
            curr = vol.select(axis, idx + 1)
            stride = max(1, int(np.ceil(max(prev.shape) / 512)))
            prev = prev[::stride, ::stride]
            curr = curr[::stride, ::stride]
            counts = transition_counts(prev, curr, num_phases)
            if idx in band_set:
                band_counts[idx] = counts
            elif idx in inner_set:
                inner_counts.add_(counts)

        if not inner_idx or not band_idx:
            tvs.append(None)
            deltas.append(None)
            continue

        axis_tv = []
        axis_delta = []
        for idx in band_idx:
            seam_counts = band_counts[idx]
            axis_tv.append(transition_tv(seam_counts, inner_counts))
            axis_delta.append(continuation_delta(seam_counts, inner_counts))
        tvs.append(max(axis_tv))
        deltas.append(max(axis_delta))
    return SeamQuality(
        transition_tv=tuple(tvs),
        continuation_delta=tuple(deltas),
    )
