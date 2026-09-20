import numpy as np
import pytest
import torch

from src.evaluate.connectivity import (
    continuation_delta,
    transition_counts,
    transition_tv,
)
from src.evaluate.seam import measure_seams


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


@pytest.mark.parametrize("previous,current", [(0, 2), (1, -1)])
def test_invalid_phase_cannot_alias_a_valid_transition(previous, current):
    with pytest.raises(ValueError, match="phases from 0 to 1"):
        transition_counts(torch.tensor([previous]), torch.tensor([current]), 2)


def test_continuation_is_unavailable_without_a_shared_source_phase():
    first = torch.tensor([[1, 0], [0, 0]])
    second = torch.tensor([[0, 0], [0, 1]])
    assert continuation_delta(first, second) is None


@pytest.mark.parametrize("metric", [transition_tv, continuation_delta])
@pytest.mark.parametrize(
    "counts",
    [
        torch.zeros(2, 2),
        torch.ones(1, 2),
        -torch.ones(2, 2),
        torch.full((2, 2), float("nan")),
    ],
)
def test_transition_metrics_reject_invalid_histograms(metric, counts):
    with pytest.raises(ValueError):
        metric(counts, torch.ones(2, 2))


def test_seams_report_missing_continuation_without_crashing():
    labels = torch.ones(10, 2, 2, dtype=torch.long)
    labels[4:6] = 0
    metrics = measure_seams(labels, ((5,), (), ()), band_size=1, num_phases=2)
    assert metrics.transition_tv[0] is not None
    assert metrics.continuation_delta[0] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_seam_accumulation_stays_on_the_volume_device():
    labels = torch.zeros(10, 2, 2, dtype=torch.long, device="cuda")
    metrics = measure_seams(labels, ((5,), (), ()), band_size=1, num_phases=2)
    assert metrics.transition_tv[0] == 0
    assert metrics.continuation_delta[0] == 0
