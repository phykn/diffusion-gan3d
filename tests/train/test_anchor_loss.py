import torch

from src.anchor import PlaneAnchor, build_anchors
from src.train.anchor_loss import soft_anchor_loss


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
        coarse_weight=1.0,
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
        coarse_weight=1.0,
        pixel_weight=0.05,
    )
    second_loss = soft_anchor_loss(
        second,
        condition,
        torch.tensor((True,)),
        pool_size=4,
        coarse_weight=1.0,
        pixel_weight=0.05,
    )

    torch.testing.assert_close(first_loss.total, second_loss.total)
    torch.testing.assert_close(first_loss.coarse, second_loss.coarse)
    torch.testing.assert_close(first_loss.pixel, second_loss.pixel)


def test_soft_anchor_loss_applies_configured_pixel_weight() -> None:
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
        coarse_weight=0.0,
        pixel_weight=0.05,
    )

    torch.testing.assert_close(result.total, 0.05 * result.pixel)


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
        coarse_weight=1.0,
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
        coarse_weight=1.0,
        pixel_weight=0.05,
    )

    assert float(result.total) < 1e-4
    assert result.visible_voxels == 16


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
