"""TensorBoard scalars of a training step and the structure metrics."""

import math
from itertools import combinations

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.phase import PORE_PHASE
from src.plane import AXIS_NAMES, PLANES
from src.stats.sections import label_sections, section_errors, two_point
from src.stats.volume import spanning_fractions, tilt_balance


def write_metrics(writer: SummaryWriter, step: int, metrics: dict) -> None:
    for tag, value in metrics.items():
        # Unavailable measurements remain visible as gaps at their recorded steps.
        if value is None:
            value = math.nan
        writer.add_scalar(tag, float(value), step)


@torch.no_grad()
def structure_metrics(probs, real: dict, groups: dict) -> dict:
    """structure/* tags of [B, K, D, H, W] predictions against real sections."""
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
            span.append(spanning_fractions(mask))
            lower.append(
                [
                    float(mask.all(axis=axis).sum() * mask.shape[axis] / total)
                    for axis in range(3)
                ]
            )
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
        images = label_sections(images.detach().cpu())
        planes = probs.movedim(axis + 2, 1).flatten(0, 1)
        indices = torch.linspace(0, len(planes) - 1, len(images)).long()
        fake = label_sections(
            planes[indices][..., : images.shape[-2], : images.shape[-1]]
        )
        for name, value in section_errors(images, fake)._asdict().items():
            metrics[f"structure/{PLANES[axis]}/{name}"] = value
        generated_curves[axis] = two_point(fake)
    for group, axes in groups.items():
        pairs = list(combinations([a for a in axes if a in generated_curves], 2))
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
                            for x, y in zip(
                                generated_curves[a], generated_curves[b], strict=True
                            )
                        ]
                    )
                    for a, b in pairs
                ]
            )
        )
    return metrics


def selection_metrics(entry: dict, best: dict) -> dict:
    """selection/* tags of one scored step."""
    values = {
        f"selection/{name}": value for name, value in entry.items() if name != "step"
    }
    values["selection/best_step"] = best["step"]
    return values
