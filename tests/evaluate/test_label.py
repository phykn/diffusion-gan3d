import numpy as np
import pytest
import torch

from src.evaluate import (
    phase_fraction,
    phase_fractions,
    voxel_accuracy,
)


def test_phase_fractions_support_multiphase_numpy_and_torch_inputs() -> None:
    labels = np.asarray((0, 1, 1, 2), dtype=np.uint8)

    assert phase_fraction(labels, phase=1) == pytest.approx(0.5)
    assert torch.equal(
        phase_fractions(torch.from_numpy(labels), num_phases=3),
        torch.tensor((0.25, 0.5, 0.25), dtype=torch.float64),
    )


def test_voxel_accuracy_compares_labels_at_each_coordinate() -> None:
    actual = torch.tensor((0, 1, 1, 0))
    expected = torch.tensor((0, 1, 0, 2))

    assert voxel_accuracy(actual, expected) == pytest.approx(0.5)
