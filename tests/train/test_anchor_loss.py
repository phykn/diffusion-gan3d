import pytest
import torch

from src.anchor import PlaneAnchor, build_anchors
from src.train.anchor_loss import pool_size_from_downsampling, soft_anchor_loss


@pytest.mark.parametrize(
    ("downsample_factor", "expected"),
    ((1, 1), (2, 2), (4, 2), (8, 4), (16, 4)),
)
def test_anchor_pool_size_uses_the_middle_encoder_scale(
    downsample_factor: int,
    expected: int,
) -> None:
    assert pool_size_from_downsampling(downsample_factor) == expected


def test_soft_anchor_loss_uses_each_anchor_plane_as_a_2d_field() -> None:
    size = 4
    anchors = (
        PlaneAnchor(torch.zeros(size, size, dtype=torch.uint8), axis=0, index=1),
        PlaneAnchor(torch.ones(size, size, dtype=torch.uint8), axis=0, index=2),
    )
    condition = _condition(anchors, size=size)
    logits = _matching_logits(condition, phases=2)

    result = soft_anchor_loss(
        logits,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.05,
    )

    assert float(result.coarse) < 1e-4
    assert float(result.pixel) < 1e-4
    assert result.visible_voxels == 2 * size**2


def test_soft_anchor_loss_ignores_values_outside_a_partial_anchor() -> None:
    image = torch.tensor(((0, 1), (1, 0)), dtype=torch.uint8)
    condition = _condition(
        (PlaneAnchor(image, axis=1, index=2, position=(1, 1)),),
        size=4,
    )
    first = torch.randn(1, 2, 4, 4, 4)
    second = first.clone()
    outside = ~condition.mask.expand_as(second)
    second[outside] = torch.randn_like(second[outside]) * 20.0

    first_loss = soft_anchor_loss(
        first,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.05,
    )
    second_loss = soft_anchor_loss(
        second,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.05,
    )

    torch.testing.assert_close(first_loss.total, second_loss.total)
    torch.testing.assert_close(first_loss.coarse, second_loss.coarse)
    torch.testing.assert_close(first_loss.pixel, second_loss.pixel)


def test_soft_anchor_loss_uses_coarse_as_the_unit_weight() -> None:
    condition = _condition(
        (PlaneAnchor(torch.zeros(4, 4, dtype=torch.uint8), axis=2, index=1),),
        size=4,
    )
    logits = torch.zeros(1, 2, 4, 4, 4)

    result = soft_anchor_loss(
        logits,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.05,
    )

    torch.testing.assert_close(result.total, result.coarse + 0.05 * result.pixel)


def test_partial_pooling_cells_are_weighted_by_observed_coverage() -> None:
    condition = _condition(
        (
            PlaneAnchor(
                torch.zeros(5, 5, dtype=torch.uint8),
                axis=0,
                index=1,
                position=(0, 0),
            ),
        ),
        size=8,
    )
    logits = _matching_logits(condition, phases=2)
    logits[:, 0, :, 4:, 4:] = -10.0
    logits[:, 1, :, 4:, 4:] = 10.0

    result = soft_anchor_loss(
        logits,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.0,
    )
    wrong_probability = torch.tensor((-10.0, 10.0)).softmax(dim=0)[0]
    wrong_cell = -wrong_probability.clamp_min(torch.finfo(torch.float32).eps).log()

    torch.testing.assert_close(result.coarse, wrong_cell / 25.0)


def test_hidden_anchor_has_differentiable_zero_loss() -> None:
    condition = _condition(
        (PlaneAnchor(torch.zeros(4, 4, dtype=torch.uint8), axis=0, index=1),),
        size=4,
    )
    logits = torch.randn(1, 2, 4, 4, 4, requires_grad=True)

    result = soft_anchor_loss(
        logits,
        condition,
        torch.tensor((False,)),
        pool_size=4,
        pixel_weight=0.05,
    )
    result.total.backward()

    assert float(result.total.detach()) == 0.0
    assert result.visible_voxels == 0
    assert logits.grad is not None
    assert not bool(logits.grad.any())


def test_visibility_selects_individual_anchor_batch_items() -> None:
    images = torch.stack(
        (
            torch.zeros(4, 4, dtype=torch.uint8),
            torch.ones(4, 4, dtype=torch.uint8),
        )
    )
    condition = build_anchors(
        (PlaneAnchor(images, axis=0, index=1),),
        batch_size=2,
        num_phases=2,
        volume_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition is not None
    logits = _matching_logits(condition, phases=2)
    logits[1].mul_(-1.0)

    result = soft_anchor_loss(
        logits,
        condition,
        torch.tensor((True, False)),
        pool_size=4,
        pixel_weight=0.05,
    )

    assert float(result.total) < 1e-4
    assert result.visible_voxels == 16


def test_pixel_loss_uses_only_the_original_observed_plane() -> None:
    size = 4
    observed = PlaneAnchor(torch.zeros(size, size, dtype=torch.uint8), axis=0, index=1)
    generated = PlaneAnchor(torch.ones(size, size, dtype=torch.uint8), axis=0, index=2)
    condition = _condition((observed, generated), size=size)
    observed_condition = _condition((observed,), size=size)
    logits = _matching_logits(observed_condition, phases=2)

    result = soft_anchor_loss(
        logits,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.05,
        observed_mask=observed_condition.mask,
        observed_axis_masks=observed_condition.axis_masks,
    )

    assert float(result.coarse) > 0.1
    assert float(result.pixel) < 1e-4
    assert result.visible_voxels == size**2


def test_observed_and_generated_coarse_groups_are_balanced() -> None:
    size = 4
    observed = PlaneAnchor(torch.zeros(size, size, dtype=torch.uint8), axis=0, index=0)
    pseudo = tuple(
        PlaneAnchor(torch.ones(size, size, dtype=torch.uint8), axis=0, index=index)
        for index in (1, 2, 3)
    )
    observed_condition = _condition((observed,), size=size)
    single = _condition((observed, pseudo[0]), size=size)
    multiple = _condition((observed, *pseudo), size=size)
    logits = _matching_logits(observed_condition, phases=2)

    single_loss = soft_anchor_loss(
        logits,
        single,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.0,
        observed_mask=observed_condition.mask,
        observed_axis_masks=observed_condition.axis_masks,
    )
    multiple_loss = soft_anchor_loss(
        logits,
        multiple,
        torch.tensor((True,)),
        pool_size=4,
        pixel_weight=0.0,
        observed_mask=observed_condition.mask,
        observed_axis_masks=observed_condition.axis_masks,
    )

    torch.testing.assert_close(single_loss.coarse, multiple_loss.coarse)


def _condition(anchors: tuple[PlaneAnchor, ...], *, size: int):
    condition = build_anchors(
        anchors,
        batch_size=1,
        num_phases=2,
        volume_size=size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition is not None
    return condition


def _matching_logits(condition, *, phases: int) -> torch.Tensor:
    logits = torch.full(
        (condition.target.shape[0], phases, *condition.target.shape[1:]),
        -10.0,
    )
    logits.scatter_(1, condition.target.unsqueeze(1), 10.0)
    return logits
