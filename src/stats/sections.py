from typing import NamedTuple

import numpy as np
import scipy.ndimage as ndi
import torch


def two_point(probs, max_lag=16):
    """Per in-plane axis, S2 per phase and lag of [B, K, H, W] sections: [K, L]."""
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


def label_sections(sections, seed: int = 0) -> torch.Tensor:
    """One-hot sections, with exact phase ties broken at random.

    Area-downsampled crops hold exact ties such as 0.5/0.5, which argmax would
    all give to the lowest phase.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(sections.shape, generator=generator) * 1e-3
    labels = (sections.float() + noise).argmax(1)
    return torch.nn.functional.one_hot(labels, sections.shape[1]).movedim(-1, 1).float()


class SectionStatistics(NamedTuple):
    two_point: list
    chords: list
    clusters: tuple


def section_statistics(sections) -> SectionStatistics:
    """The statistics compare_sections uses, computed once for fixed sections."""
    phases = sections.shape[1]
    labels = sections.argmax(1).numpy()
    return SectionStatistics(
        two_point(sections),
        chord_distribution(labels, phases),
        cluster_statistics(labels, phases),
    )


def compare_sections(real: SectionStatistics, fake: SectionStatistics) -> SectionErrors:
    two_point_mae = float(
        torch.cat(
            [(a - b).abs().flatten() for a, b in zip(real.two_point, fake.two_point)]
        ).mean()
    )
    chord_tv = float(
        np.mean(
            [
                np.abs(a - b).sum(-1).mean() * 0.5
                for a, b in zip(real.chords, fake.chords)
            ]
        )
    )
    # Gap relative to the mean of both sides: bounded at 2 when one side lacks a
    # phase, and 0 when neither has it.
    gaps = []
    for a, b in zip(real.clusters, fake.clusters):
        mean = (a + b) / 2
        gaps.append(
            np.divide(np.abs(a - b), mean, out=np.zeros_like(mean), where=mean > 0)
        )
    cluster_error = float(np.mean(gaps))
    return SectionErrors(two_point_mae, chord_tv, cluster_error)


def section_errors(real, fake) -> SectionErrors:
    """How far generated sections are from real ones in three statistics."""
    return compare_sections(section_statistics(real), section_statistics(fake))


def half_split_errors(sections) -> list[float]:
    """Errors between the even and odd halves of real one-hot sections.

    The sampling noise of the real data puts every error on one scale.
    """
    return [
        max(error, 1e-6) for error in section_errors(sections[0::2], sections[1::2])
    ]
