import numpy as np
import pytest
import torch

from src.evaluate import (
    continuation_delta,
    continuation_error,
    phase_change_rate,
    phase_continuation,
    transition_counts,
    transition_tv,
)


def test_transition_metrics_count_multiphase_pairs() -> None:
    previous = np.asarray(((0, 0, 1, 1), (0, 1, 2, 2)), dtype=np.uint8)
    current = np.asarray(((0, 1, 1, 2), (2, 1, 2, 0)), dtype=np.uint8)

    counts = transition_counts(previous, current, num_phases=3)

    assert torch.equal(
        counts,
        torch.tensor(
            ((1, 1, 1), (0, 2, 1), (1, 0, 1)),
            dtype=torch.float64,
        ),
    )
    assert phase_change_rate(counts) == pytest.approx(0.5)
    assert phase_continuation(counts).tolist() == pytest.approx((1 / 3, 2 / 3, 1 / 2))


def test_transition_comparisons_report_distribution_and_continuation_gaps() -> None:
    first = torch.tensor(((3.0, 1.0), (1.0, 1.0)))
    second = torch.tensor(((1.0, 3.0), (0.0, 2.0)))

    assert transition_tv(first, second) == pytest.approx(0.5)
    assert continuation_delta(first, second) == pytest.approx(0.5)
    assert phase_continuation(first).tolist() == pytest.approx((0.75, 0.5))
    assert continuation_error(
        torch.tensor((0.75, 0.0)),
        torch.tensor((0.80, 0.0)),
        phase=0,
    ) == pytest.approx(0.05)


def test_phase_continuation_rejects_phase_without_source_support() -> None:
    counts = torch.tensor(((3.0, 1.0), (0.0, 0.0)))

    with pytest.raises(ValueError, match=r"missing phases: \[1\]"):
        phase_continuation(counts)


def test_continuation_delta_uses_only_shared_supported_phases() -> None:
    first = torch.tensor(((3.0, 1.0), (1.0, 1.0)))
    phase_zero_only = torch.tensor(((1.0, 3.0), (0.0, 0.0)))

    assert continuation_delta(first, phase_zero_only) == pytest.approx(0.5)

    phase_one_only = torch.tensor(((0.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="phase supported in both"):
        continuation_delta(phase_zero_only, phase_one_only)


def test_transition_counts_reject_labels_outside_num_phases() -> None:
    with pytest.raises(ValueError, match="phases from 0 to 1"):
        transition_counts(torch.tensor((0, 2)), torch.tensor((0, 1)), 2)
