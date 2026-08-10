import numpy as np
import pytest
import torch

from src.evaluate import (
    phase_fraction,
    phase_fractions,
    phase_iou,
    phase_recall,
    voxel_accuracy,
)


def test_phase_fractions_support_multiphase_numpy_and_torch_inputs() -> None:
    labels = np.asarray((0, 1, 1, 2), dtype=np.uint8)

    assert phase_fraction(labels, phase=1) == pytest.approx(0.5)
    assert torch.equal(
        phase_fractions(torch.from_numpy(labels), num_phases=3),
        torch.tensor((0.25, 0.5, 0.25), dtype=torch.float64),
    )


def test_label_comparison_metrics_preserve_empty_phase_convention() -> None:
    actual = torch.tensor((0, 1, 1, 0))
    expected = torch.tensor((0, 1, 0, 2))

    assert voxel_accuracy(actual, expected) == pytest.approx(0.5)
    assert phase_iou(actual, expected, num_phases=4) == pytest.approx(
        (1 / 3, 1 / 2, 0.0, 1.0)
    )
    assert phase_recall(actual, expected, num_phases=4) == pytest.approx(
        (1 / 2, 1.0, 0.0, 1.0)
    )


def test_label_comparison_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        voxel_accuracy(np.zeros((2, 2)), np.zeros((2, 3)))


@pytest.mark.parametrize("metric", (phase_iou, phase_recall))
@pytest.mark.parametrize(
    ("actual", "expected"),
    (
        (torch.tensor((0, 2)), torch.tensor((0, 1))),
        (torch.tensor((0, 1)), torch.tensor((0, 2))),
    ),
)
def test_label_comparison_rejects_labels_outside_num_phases(
    metric,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="phases from 0 to 1"):
        metric(actual, expected, num_phases=2)
