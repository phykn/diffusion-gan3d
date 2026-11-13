from dataclasses import dataclass

import numpy as np
import torch

from src.device import wide_dtype
from src.stats.labels import (
    continuation_delta,
    transition_counts,
    transition_tv,
)


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
