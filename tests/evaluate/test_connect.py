import numpy as np
import pytest
import torch

from src.evaluate import (
    continuation_delta,
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
def test_transition_comparisons_report_distribution_and_continuation_gaps() -> None:
    first = torch.tensor(((3.0, 1.0), (1.0, 1.0)))
    second = torch.tensor(((1.0, 3.0), (0.0, 2.0)))

    assert transition_tv(first, second) == pytest.approx(0.5)
    assert continuation_delta(first, second) == pytest.approx(0.5)


def test_transition_counts_reject_labels_outside_num_phases() -> None:
    with pytest.raises(ValueError, match="phases from 0 to 1"):
        transition_counts(torch.tensor((0, 2)), torch.tensor((0, 1)), 2)
