import math

import pytest
import torch

from src.train.scale_consistency import (
    cosine_overlap_weights,
    make_adjacent_view_plan,
    periodic_crop_3d,
    weighted_probability_mse,
)


def coordinate_volume(size: int) -> torch.Tensor:
    return torch.arange(size**3).reshape(1, 1, size, size, size)


def test_periodic_crop_identity_preserves_values_and_gradients() -> None:
    values = torch.randn(1, 2, 3, 4, 5, requires_grad=True)

    cropped = periodic_crop_3d(values, (0, 0, 0), (3, 4, 5))
    cropped.sum().backward()

    assert torch.equal(cropped, values)
    assert values.grad is not None
    assert torch.equal(values.grad, torch.ones_like(values))


def test_periodic_crop_supports_negative_offsets_and_repeated_wraps() -> None:
    values = coordinate_volume(3)

    cropped = periodic_crop_3d(values, (-1, 2, 4), (5, 4, 3))
    expected = values
    for dim, indices in zip(
        (-3, -2, -1),
        (
            torch.tensor((2, 0, 1, 2, 0)),
            torch.tensor((2, 0, 1, 2)),
            torch.tensor((1, 2, 0)),
        ),
        strict=True,
    ):
        expected = expected.index_select(dim, indices)

    assert torch.equal(cropped, expected)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_adjacent_view_bands_reference_the_same_global_voxels(axis: int) -> None:
    plan = make_adjacent_view_plan(
        origin=(3, 5, 7),
        axis=axis,
        core_size=4,
        overlap=2,
    )
    first, second = plan.crop_views(coordinate_volume(16))
    first_band, second_band = plan.overlap_bands(first, second)

    assert plan.tile_size == 8
    assert first_band.shape[axis + 2] == second_band.shape[axis + 2] == 4
    assert torch.equal(first_band, second_band)


def test_adjacent_view_contract_remains_aligned_across_periodic_boundary() -> None:
    plan = make_adjacent_view_plan(
        origin=(0, 0, 0),
        axis=0,
        core_size=4,
        overlap=2,
    )
    first, second = plan.crop_views(coordinate_volume(6))

    first_band, second_band = plan.overlap_bands(first, second)

    assert plan.first_start == (-2, -2, -2)
    assert plan.second_start == (2, -2, -2)
    assert torch.equal(first_band, second_band)


def test_cosine_weights_match_the_taper_product() -> None:
    weights = cosine_overlap_weights(2, dtype=torch.float64)

    torch.testing.assert_close(
        weights,
        torch.tensor((0.0, 0.5, 0.5, 0.0), dtype=torch.float64),
    )
    assert math.isclose(float(weights.sum()), 1.0)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_weighted_probability_mse_uses_the_selected_band_axis(axis: int) -> None:
    shape = [1, 2, 3, 3, 3]
    shape[axis + 2] = 4
    teacher = torch.zeros(shape)
    student = torch.ones(shape)

    loss = weighted_probability_mse(
        student,
        teacher,
        axis=axis,
        overlap=2,
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_zero_weight_endpoints_do_not_contribute() -> None:
    teacher = torch.zeros(1, 2, 4, 2, 2)
    student = teacher.clone()
    student[:, :, (0, 3)] = 1.0

    loss = weighted_probability_mse(
        student,
        teacher,
        axis=0,
        overlap=2,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_nonzero_weight_anchor_position_contributes_to_loss() -> None:
    teacher = torch.zeros(1, 2, 4, 2, 2)
    student = teacher.clone()
    student[:, :, 1, 0, 0] = 1.0

    loss = weighted_probability_mse(
        student,
        teacher,
        axis=0,
        overlap=2,
    )

    torch.testing.assert_close(loss, torch.tensor(0.125))


def test_teacher_is_stop_gradient_and_student_receives_gradient() -> None:
    student = torch.rand(1, 2, 4, 2, 2, requires_grad=True)
    teacher = torch.rand(1, 2, 4, 2, 2, requires_grad=True)

    loss = weighted_probability_mse(
        student,
        teacher,
        axis=0,
        overlap=2,
    )
    loss.backward()

    assert student.grad is not None
    assert bool((student.grad != 0).any())
    assert teacher.grad is None


@pytest.mark.parametrize(
    ("call", "error"),
    (
        (lambda: periodic_crop_3d(torch.zeros(2, 2), (0, 0, 0), 2), "three"),
        (lambda: periodic_crop_3d(torch.zeros(2, 2, 2), (0, 0), 2), "three"),
        (lambda: periodic_crop_3d(torch.zeros(2, 2, 2), (0, 0, 0), 0), "positive"),
        (
            lambda: make_adjacent_view_plan((0, 0, 0), 3, 4, 2),
            "axis",
        ),
        (
            lambda: make_adjacent_view_plan((0, 0, 0), 0, 4, 1),
            "overlap",
        ),
        (
            lambda: weighted_probability_mse(
                torch.zeros(1, 2, 3, 2, 2),
                torch.zeros(1, 2, 3, 2, 2),
                axis=0,
                overlap=2,
            ),
            r"2 \* overlap",
        ),
    ),
)
def test_invalid_geometry_is_rejected(call, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        call()
