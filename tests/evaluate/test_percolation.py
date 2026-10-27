import numpy as np
import pytest

from src.evaluate.connectivity import spanning_fractions


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_spanning_fractions_handle_each_axis(axis: int) -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    selector = [2, 2, 2]
    selector[axis] = slice(None)
    mask[tuple(selector)] = True

    expected = [0.0, 0.0, 0.0]
    expected[axis] = 1.0
    assert spanning_fractions(mask) == pytest.approx(expected)


def test_spanning_fractions_sum_all_spanning_components() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, 0, 0] = True
    mask[:, 2, 2] = True
    mask[1, 1, 1] = True

    assert spanning_fractions(mask) == pytest.approx((6 / 7, 0.0, 0.0))


def test_nonspanning_components_return_zero() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 1, 1] = True
    mask[2, 1, 1] = True

    assert spanning_fractions(mask) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "points",
    (
        ((0, 0, 0), (1, 1, 1), (2, 2, 2)),
        ((0, 0, 1), (1, 1, 1), (2, 2, 1)),
    ),
    ids=("corner", "edge"),
)
def test_corner_and_edge_contacts_do_not_connect(points) -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    for point in points:
        mask[point] = True

    assert spanning_fractions(mask) == (0.0, 0.0, 0.0)


def test_empty_mask_spans_nothing() -> None:
    assert spanning_fractions(np.zeros((3, 3, 3), dtype=bool)) == (0.0, 0.0, 0.0)
