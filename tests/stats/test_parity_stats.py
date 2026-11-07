"""src/stats against the v1 reference (SPEC Parity checks: stats)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.stats.anchor import (
    measure_boundaries,
    measure_distance_divergence,
    measure_slice_smoothness,
)
from src.stats.pairs import (
    mirror_differences,
    pair_correlations,
    section_pairs,
    volume_pairs,
)
from src.stats.seam import measure_seams
from src.stats.sections import (
    chord_distribution,
    cluster_statistics,
    compare_sections,
    half_split_errors,
    label_sections,
    section_statistics,
    two_point,
)
from src.stats.volume import spanning_fractions, tilt_balance, tortuosity

REFERENCE = torch.load(
    Path(__file__).parents[1] / "fixtures" / "parity" / "stats.pt", weights_only=False
)
ATOL = 1e-6


def _close(actual, expected, atol=ATOL):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _close(actual[key], expected[key], atol)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _close(a, b, atol)
    elif expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, abs=atol)


def test_pair_tables():
    pairs = REFERENCE["pairs"]
    images, volume = pairs["images"], pairs["volume"]
    for (dim, gap), expected in pairs["section"].items():
        _close(section_pairs(images, dim, gap), expected)
    for (axis, gap, keep), expected in pairs["volume_pairs"].items():
        _close(volume_pairs(volume, axis, gap, keep), expected)
    correlations = pair_correlations(volume, 4)
    _close(correlations, pairs["correlations"], 1e-5)
    for (first, second), expected in pairs["mirror"].items():
        offsets = [(1, 1), (1, 2), (2, 1)]
        actual = mirror_differences(
            correlations, volume.shape[2:], first, second, offsets
        )
        _close(actual, expected)


def test_section_statistics():
    sections = REFERENCE["sections"]
    real = sections["real"]
    labels = real.argmax(1).numpy()
    _close(two_point(real), sections["two_point"])
    _close(chord_distribution(labels, 3), sections["chords"])
    _close(cluster_statistics(labels, 3), sections["clusters"])
    fake = label_sections(sections["fake_probs"], seed=3)
    _close(fake, sections["fake"])
    errors = compare_sections(section_statistics(real), section_statistics(fake))
    _close(tuple(errors), sections["compare"])
    _close(half_split_errors(real), sections["half_split"])


def test_volume_statistics():
    volumes = REFERENCE["volumes"]
    labels = volumes["labels"]
    _close(
        [spanning_fractions(labels.numpy() == phase) for phase in (0, 1)],
        volumes["spanning"],
    )
    _close(
        [tortuosity(labels, phase=0, axis=axis, device="cpu") for axis in range(3)],
        volumes["tortuosity"],
        1e-3,
    )
    _close(tilt_balance([volumes["tilt_volume"].numpy()], phase=1), volumes["tilt"])


def test_anchor_metrics():
    anchor = REFERENCE["anchor"]
    anchored, baseline, indices = (
        anchor["anchored"],
        anchor["baseline"],
        anchor["indices"],
    )
    for axis in range(3):
        _close(
            tuple(vars(measure_boundaries(anchored, indices, axis)).values()),
            anchor["boundaries"][axis],
        )
        _close(
            measure_distance_divergence(anchored, baseline, indices, axis),
            anchor["divergence"][axis],
        )
        smoothness = measure_slice_smoothness(anchored, indices, axis, baseline)
        _close(tuple(vars(smoothness).values()), anchor["smoothness"][axis])


def test_seams():
    seam = REFERENCE["seam"]
    quality = measure_seams(seam["volume"], seam["seams"], 3, 3)
    _close(quality.transition_tv, seam["transition_tv"])
    _close(quality.continuation_delta, seam["continuation_delta"])
