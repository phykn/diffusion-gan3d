import subprocess
import sys

import numpy as np
import pytest
import torch

from src.evaluate.label import phase_fraction, phase_fractions, voxel_accuracy


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


def test_voxel_accuracy_rejects_broadcasting():
    with pytest.raises(ValueError, match="same shape"):
        voxel_accuracy(torch.zeros(2, 3), torch.zeros(3))


@pytest.mark.parametrize(
    "values", [[0, 2], [0, 0.5], [], np.array([2**63], dtype=np.uint64)]
)
def test_phase_fractions_reject_invalid_or_empty_labels(values):
    with pytest.raises(ValueError):
        phase_fractions(values, 2)


def test_label_metrics_do_not_import_inception_or_transport_solvers():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import src.evaluate.label; "
            "assert 'torchmetrics' not in sys.modules; "
            "assert 'taufactor' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
