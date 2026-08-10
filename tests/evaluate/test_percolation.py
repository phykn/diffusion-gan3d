import numpy as np
import pytest
import torch

from src.evaluate import (
    percolating_fraction,
    percolating_fractions,
    percolation_error,
    percolation_errors,
)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_percolating_fractions_handle_each_axis(axis: int) -> None:
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    selector = [2, 2, 2]
    selector[axis] = slice(None)
    volume[tuple(selector)] = 1

    fractions = percolating_fractions(volume, phase=1)

    expected = [0.0, 0.0, 0.0]
    expected[axis] = 1.0
    assert fractions == pytest.approx(expected)
    assert percolating_fraction(volume, phase=1, axis=axis) == 1.0


def test_percolating_fractions_support_phase_zero() -> None:
    volume = np.ones((4, 4, 4), dtype=np.uint8)
    volume[:, 2, 2] = 0

    assert percolating_fractions(volume, phase=0) == (1.0, 0.0, 0.0)


def test_percolating_fraction_sums_all_spanning_components() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[:, 0, 0] = 1
    volume[:, 2, 2] = 1
    volume[1, 1, 1] = 1

    assert percolating_fractions(volume) == pytest.approx((6 / 7, 0.0, 0.0))


def test_nonpercolating_components_return_zero() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[0, 1, 1] = 1
    volume[2, 1, 1] = 1

    assert percolating_fractions(volume) == (0.0, 0.0, 0.0)


def test_corner_contact_does_not_connect_components() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[0, 0, 0] = 1
    volume[1, 1, 1] = 1
    volume[2, 2, 2] = 1

    assert percolating_fractions(volume) == (0.0, 0.0, 0.0)


def test_edge_contact_does_not_connect_components() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[0, 0, 1] = 1
    volume[1, 1, 1] = 1
    volume[2, 2, 1] = 1

    assert percolating_fractions(volume) == (0.0, 0.0, 0.0)


def test_percolation_error_is_axiswise_before_averaging() -> None:
    predicted = (1.0, 0.0, 0.0)
    target = (0.0, 1.0, 0.0)

    assert percolation_errors(predicted, target) == (1.0, 1.0, 0.0)
    assert percolation_error(predicted, target) == pytest.approx(2 / 3)


def test_identical_gt_and_prediction_have_zero_percolation_error() -> None:
    volume = np.ones((2, 3, 4), dtype=np.uint8)
    fractions = percolating_fractions(volume)

    assert fractions == (1.0, 1.0, 1.0)
    assert percolation_errors(fractions, fractions) == (0.0, 0.0, 0.0)
    assert percolation_error(fractions, fractions) == 0.0


def test_percolation_rejects_absent_phase() -> None:
    with pytest.raises(ValueError, match="no voxels for phase 1"):
        percolating_fractions(np.zeros((3, 3, 3), dtype=np.uint8), phase=1)

    with pytest.raises(ValueError, match="no voxels for phase 0"):
        percolating_fractions(np.ones((3, 3, 3), dtype=np.uint8), phase=0)


def test_percolation_preserves_torch_input() -> None:
    volume = torch.zeros((3, 3, 3), dtype=torch.uint8)
    volume[:, 1, 1] = 1
    original = volume.clone()

    assert percolating_fractions(volume) == (1.0, 0.0, 0.0)
    assert torch.equal(volume, original)


@pytest.mark.parametrize(
    "volume",
    (
        np.ones((3, 3), dtype=np.uint8),
        np.ones((1, 3, 3), dtype=np.uint8),
    ),
)
def test_percolation_rejects_invalid_volume_shape(volume: np.ndarray) -> None:
    with pytest.raises(ValueError, match="3D array|at least two voxels"):
        percolating_fractions(volume)


def test_percolation_error_rejects_invalid_fraction_vectors() -> None:
    with pytest.raises(ValueError, match="three axes"):
        percolation_error((1.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        percolation_error((1.1, 0.0, 0.0), (1.0, 0.0, 0.0))
