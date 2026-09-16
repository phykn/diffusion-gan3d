"""Reference-free 3D diagnostics and comparisons against measured 2D slices."""

from itertools import combinations

import numpy as np
import scipy.ndimage as ndi
import torch

from src.plane import PLANES


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
            components, _ = ndi.label(
                mask, structure=ndi.generate_binary_structure(3, 1)
            )
            counts = np.bincount(components.ravel())
            total = max(int(mask.sum()), 1)
            fractions, bounds = [], []
            for axis in range(3):
                connected = np.intersect1d(
                    np.take(components, 0, axis), np.take(components, -1, axis)
                )
                connected = connected[connected != 0]
                fractions.append(float(counts[connected].sum() / total))
                bounds.append(
                    float(mask.all(axis=axis).sum() * mask.shape[axis] / total)
                )
            span.append(fractions)
            lower.append(bounds)
        percolation[phase] = np.mean(span, axis=0)
        for axis in range(3):
            metrics[f"structure/phase_{phase}/percolation_{PLANES[axis]}"] = (
                percolation[phase, axis]
            )
            metrics[
                f"structure/phase_{phase}/percolation_lower_bound_{PLANES[axis]}"
            ] = float(np.mean(lower, axis=0)[axis])
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
        first, second = two_point(images), two_point(fake)
        metrics[f"structure/{PLANES[axis]}/two_point_mae"] = float(
            torch.cat([(a - b).abs().flatten() for a, b in zip(first, second)]).mean()
        )
        first_chords = chord_distribution(images.argmax(1).numpy(), phases)
        second_chords = chord_distribution(fake.argmax(1).numpy(), phases)
        metrics[f"structure/{PLANES[axis]}/chord_tv"] = float(
            np.mean(
                [
                    np.abs(a - b).sum(-1).mean() * 0.5
                    for a, b in zip(first_chords, second_chords)
                ]
            )
        )
        generated_curves[axis] = second
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
