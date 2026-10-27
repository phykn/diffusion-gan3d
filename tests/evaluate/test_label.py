import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import src.evaluate.label as label_module
from src.evaluate.label import phase_fractions, voxel_accuracy


def test_phase_fractions_support_multiphase_numpy_and_torch_inputs() -> None:
    labels = np.asarray((0, 1, 1, 2), dtype=np.uint8)

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


def test_label_metrics_do_not_import_transport_solvers():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import src.evaluate.label; "
            "assert 'taufactor' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), float("inf")])
def test_chunked_phase_fractions_still_validate_every_label(monkeypatch, value):
    monkeypatch.setattr(label_module, "LABEL_CHUNK_ELEMENTS", 4)
    values = torch.zeros(3, 5)
    values[0, 0] = value
    with pytest.raises(ValueError):
        phase_fractions(values, 2)


def test_chunked_phase_fractions_support_scalars_and_reject_empty_input():
    assert phase_fractions(torch.tensor(1), 2).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="non-empty"):
        phase_fractions(torch.empty(0), 2)


@pytest.mark.parametrize("strided", [False, True])
def test_phase_fractions_bounds_intermediate_allocations(monkeypatch, strided):
    values = (torch.arange(7 * 9 * 11).reshape(7, 9, 11) % 3).to(torch.uint8)
    if strided:
        values = values.transpose(0, 2)
    expected = torch.tensor(
        [(values == phase).sum().item() / values.numel() for phase in range(3)],
        dtype=torch.float64,
    )
    monkeypatch.setattr(label_module, "LABEL_CHUNK_ELEMENTS", 32)

    class BoundedAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if str(func) in ("aten._to_copy.default", "aten.clone.default"):
                assert result.numel() <= 32
            return result

    with BoundedAllocations():
        torch.testing.assert_close(phase_fractions(values, 3), expected)


@pytest.mark.parametrize("value", [-1, 3, 0.5, float("nan"), float("inf")])
def test_chunked_phase_fractions_validates_every_chunk(monkeypatch, value):
    monkeypatch.setattr(label_module, "LABEL_CHUNK_ELEMENTS", 4)
    values = torch.zeros(3, 5)
    values[0, 0] = value
    with pytest.raises(ValueError):
        phase_fractions(values, 3)
