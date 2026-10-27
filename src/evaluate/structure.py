from itertools import combinations
from typing import NamedTuple

import numpy as np
import scipy.ndimage as ndi
import torch

from src.evaluate.connectivity import spanning_fractions
from src.evaluate.tilt import tilt_balance
from src.phase import PORE_PHASE
from src.plane import AXIS_NAMES, PLANES


def two_point(probs, max_lag=16):
    values = []
    for axis in (-2, -1):
        curve = []
        for lag in range(min(max_lag + 1, probs.shape[axis])):
            if lag == 0:
                curve.append(probs.mean((0, 2, 3)))
            else:
                left, right = [slice(None)] * 4, [slice(None)] * 4
                left[axis], right[axis] = slice(None, -lag), slice(lag, None)
                curve.append((probs[tuple(left)] * probs[tuple(right)]).mean((0, 2, 3)))
        values.append(torch.stack(curve, -1))
    return values


def chord_distribution(labels, phases):
    result = []
    for axis in (-2, -1):
        lines = np.moveaxis(labels, axis, -1).reshape(-1, labels.shape[axis])
        curves = []
        for phase in range(phases):
            binary = np.pad(lines == phase, ((0, 0), (1, 1))).astype(np.int8)
            change = np.diff(binary, axis=1)
            starts = np.where(change == 1)[1]
            ends = np.where(change == -1)[1]
            counts = np.bincount(ends - starts, minlength=lines.shape[-1] + 1)[1:]
            curves.append(counts / max(counts.sum(), 1))
        result.append(np.stack(curves))
    return result


def cluster_statistics(labels, phases):
    """Per phase: 4-connected clusters per section pixel and mean cluster area.

    Sections show how a phase splits into separate regions, which two-point and
    chord statistics miss: merged pores and fragmented solid can match both.
    """
    pixels = labels.size
    density, area = np.zeros(phases), np.zeros(phases)
    for phase in range(phases):
        mask = labels == phase
        count = sum(ndi.label(section)[1] for section in mask)
        density[phase] = count / pixels
        area[phase] = mask.sum() / max(count, 1)
    return density, area


class SectionErrors(NamedTuple):
    two_point_mae: float
    chord_tv: float
    cluster_error: float


def section_errors(real, fake) -> SectionErrors:
    """How far generated sections are from real ones in three statistics."""
    phases = real.shape[1]
    first, second = two_point(real), two_point(fake)
    two_point_mae = float(
        torch.cat([(a - b).abs().flatten() for a, b in zip(first, second)]).mean()
    )
    real_labels, fake_labels = real.argmax(1).numpy(), fake.argmax(1).numpy()
    first_chords = chord_distribution(real_labels, phases)
    second_chords = chord_distribution(fake_labels, phases)
    chord_tv = float(
        np.mean(
            [
                np.abs(a - b).sum(-1).mean() * 0.5
                for a, b in zip(first_chords, second_chords)
            ]
        )
    )
    # Gap relative to the mean of both sides: bounded at 2 when one side lacks a
    # phase, and 0 when neither has it.
    gaps = []
    for a, b in zip(
        cluster_statistics(real_labels, phases),
        cluster_statistics(fake_labels, phases),
    ):
        mean = (a + b) / 2
        gaps.append(
            np.divide(np.abs(a - b), mean, out=np.zeros_like(mean), where=mean > 0)
        )
    cluster_error = float(np.mean(gaps))
    return SectionErrors(two_point_mae, chord_tv, cluster_error)


@torch.no_grad()
def structure_metrics(probs, real, groups):
    # One scheduled transfer; never called from a diffusion or critic inner loop.
    probs = probs.detach().float().cpu()
    phases = probs.shape[1]
    labels = probs.argmax(1).numpy()
    metrics = {}
    percolation = np.zeros((phases, 3))
    for phase in range(phases):
        span, lower = [], []
        for volume in labels:
            mask = volume == phase
            total = max(int(mask.sum()), 1)
            fractions = spanning_fractions(mask)
            bounds = []
            for axis in range(3):
                bounds.append(
                    float(mask.all(axis=axis).sum() * mask.shape[axis] / total)
                )
            span.append(fractions)
            lower.append(bounds)
        percolation[phase] = np.mean(span, axis=0)
        # Percolation is transport along an axis, named z/y/x.
        for axis, name in enumerate(AXIS_NAMES):
            metrics[f"structure/phase_{phase}/percolation_{name}"] = percolation[
                phase, axis
            ]
            metrics[f"structure/phase_{phase}/percolation_lower_bound_{name}"] = float(
                np.mean(lower, axis=0)[axis]
            )
        if phase == PORE_PHASE:
            continue
        # Per-particle tilt sign; a mirror-symmetric structure stays near zero.
        for name, stats in tilt_balance(labels, phase).items():
            metrics[f"structure/phase_{phase}/tilt_balance_{name}"] = stats["balance"]
    generated_curves = {}
    for axis, images in real.items():
        images = images.detach().float().cpu()
        if images.ndim == 3:
            images = (
                torch.nn.functional.one_hot(images.long(), phases)
                .movedim(-1, 1)
                .float()
            )
        planes = probs.movedim(axis + 2, 1).flatten(0, 1)
        indices = torch.linspace(0, len(planes) - 1, len(images)).long()
        fake = planes[indices]
        fake = fake[..., : images.shape[-2], : images.shape[-1]]
        for name, value in section_errors(images, fake)._asdict().items():
            metrics[f"structure/{PLANES[axis]}/{name}"] = value
        generated_curves[axis] = two_point(fake)
    for group, axes in groups.items():
        pairs = list(combinations([a for a in axes if a in generated_curves], 2))
        if pairs:
            metrics[f"structure/{group}/percolation_axis_gap"] = float(
                np.mean(
                    [
                        np.abs(percolation[:, a] - percolation[:, b]).mean()
                        for a, b in pairs
                    ]
                )
            )
            metrics[f"structure/{group}/two_point_axis_gap"] = float(
                np.mean(
                    [
                        np.mean(
                            [
                                float((x - y).abs().mean())
                                for x, y in zip(
                                    generated_curves[a], generated_curves[b]
                                )
                            ]
                        )
                        for a, b in pairs
                    ]
                )
            )
    return metrics
