import numpy as np
import pytest
import torch

from src.evaluate import (
    percolating_fractions,
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


def test_percolating_fractions_support_phase_zero() -> None:
    volume = np.ones((4, 4, 4), dtype=np.uint8)
    volume[:, 2, 2] = 0

    assert percolating_fractions(volume, phase=0) == (1.0, 0.0, 0.0)


def test_percolating_fractions_sum_all_spanning_components() -> None:
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
