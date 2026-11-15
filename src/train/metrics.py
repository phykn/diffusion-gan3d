import math
from itertools import combinations

import numpy as np
import torch

from src.axes import AXIS_NAMES, PLANES, PORE_PHASE
from src.stats.sections import label_sections, section_errors, two_point
from src.stats.volume import spanning_fractions, tilt_balance


def write_metrics(writer, step, metrics):
    for tag, value in metrics.items():
        # Unavailable measurements remain visible as gaps at their recorded steps.
        if value is None:
            value = math.nan
        writer.add_scalar(tag, float(value), step)


@torch.no_grad()
def structure_metrics(probs, real, groups):
    """structure/* tags of [B, K, D, H, W] predictions against real sections."""
    probs = probs.detach().float().cpu()
    labels = probs.argmax(1).numpy()
    percolation, metrics = measure_percolation(labels, probs.shape[1])
    curves, sections = measure_sections(probs, real)
    return {
        **metrics,
        **measure_tilt(labels, probs.shape[1]),
        **sections,
        **measure_axis_gaps(percolation, curves, groups),
    }


def measure_percolation(labels, phases):
    """Spanning fraction of each phase along z/y/x."""
    metrics = {}
    percolation = np.zeros((phases, 3))
    for phase in range(phases):
        span, lower = [], []
        for volume in labels:
            mask = volume == phase
            total = max(int(mask.sum()), 1)
            span.append(spanning_fractions(mask))
            lower.append(
                [
                    float(mask.all(axis=axis).sum() * mask.shape[axis] / total)
                    for axis in range(3)
                ]
            )
        percolation[phase] = np.mean(span, axis=0)
        lower = np.mean(lower, axis=0)
        for axis, name in enumerate(AXIS_NAMES):
            prefix = f"structure/phase_{phase}/percolation"
            metrics[f"{prefix}_{name}"] = percolation[phase, axis]
            metrics[f"{prefix}_lower_bound_{name}"] = float(lower[axis])
    return percolation, metrics


def measure_tilt(labels, phases):
    """Per-particle tilt sign of each solid phase; a mirror-symmetric structure
    stays near zero."""
    metrics = {}
    for phase in range(phases):
        if phase == PORE_PHASE:
            continue
        for name, stats in tilt_balance(labels, phase).items():
            metrics[f"structure/phase_{phase}/tilt_balance_{name}"] = stats["balance"]
    return metrics


def measure_sections(probs, real):
    """Section errors per plane against the real crops, and the generated
    two-point curves."""
    metrics = {}
    curves = {}
    for axis, images in real.items():
        images = label_sections(images.detach().cpu())
        planes = probs.movedim(axis + 2, 1).flatten(0, 1)
        indices = torch.linspace(0, len(planes) - 1, len(images)).long()
        fake = label_sections(
            planes[indices][..., : images.shape[-2], : images.shape[-1]]
        )
        for name, value in section_errors(images, fake)._asdict().items():
            metrics[f"structure/{PLANES[axis]}/{name}"] = value
        curves[axis] = two_point(fake)
    return curves, metrics


def measure_axis_gaps(percolation, curves, groups):
    """How far the axes judged by one critic differ in percolation and two-point
    correlation."""
    metrics = {}
    for group, axes in groups.items():
        pairs = list(combinations([a for a in axes if a in curves], 2))
        if not pairs:
            continue
        metrics[f"structure/{group}/percolation_axis_gap"] = float(
            np.mean(
                [np.abs(percolation[:, a] - percolation[:, b]).mean() for a, b in pairs]
            )
        )
        metrics[f"structure/{group}/two_point_axis_gap"] = float(
            np.mean(
                [
                    np.mean(
                        [
                            float((x - y).abs().mean())
                            for x, y in zip(curves[a], curves[b], strict=True)
                        ]
                    )
                    for a, b in pairs
                ]
            )
        )
    return metrics


def selection_metrics(entry, best):
    """selection/* tags of one scored step."""
    values = {
        f"selection/{name}": value for name, value in entry.items() if name != "step"
    }
    values["selection/best_step"] = best["step"]
    return values
