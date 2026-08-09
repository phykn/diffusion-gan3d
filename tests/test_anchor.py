import pytest
import torch

from src.anchor import PlaneAnchor, build_anchors


def test_plane_anchor_builds_dense_condition_for_every_axis() -> None:
    size = 4
    image = torch.arange(size * size).reshape(size, size).remainder(3).long()

    for axis in (0, 1, 2):
        condition = build_anchors(
            (PlaneAnchor(image=image, axis=axis, index=2),),
            batch_size=1,
            num_phases=3,
            volume_size=size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert condition is not None
        assert condition.image.shape == (1, 3, size, size, size)
        assert condition.mask.shape == (1, 1, size, size, size)
        assert condition.axis_masks.shape == (1, 3, size, size, size)
        assert int(condition.mask.sum()) == size * size
        assert torch.equal(
            condition.mask, condition.axis_masks.any(dim=1, keepdim=True)
        )
        assert int(condition.axis_masks[:, axis].sum()) == size * size
        assert int(condition.axis_masks.sum()) == size * size
        assert condition.planes == 1
        assert condition.conflicts == 0
        generated = condition.image.argmax(dim=1).select(axis + 1, 2)
        assert torch.equal(generated[0], image)


def test_cross_axis_anchor_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting intersections"):
        build_anchors(
            (
                PlaneAnchor(
                    image=torch.zeros(4, 4, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
                PlaneAnchor(
                    image=torch.ones(4, 4, dtype=torch.long),
                    axis=1,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=3,
            volume_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_cross_axis_training_anchors_reconcile_intersections() -> None:
    size = 4
    condition = build_anchors(
        (
            PlaneAnchor(
                image=torch.zeros(size, size, dtype=torch.long),
                axis=0,
                index=1,
            ),
            PlaneAnchor(
                image=torch.ones(size, size, dtype=torch.long),
                axis=1,
                index=1,
            ),
        ),
        batch_size=1,
        num_phases=3,
        volume_size=size,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reconcile=True,
    )

    assert condition is not None
    assert condition.planes == 2
    assert condition.conflicts == size
    assert condition.source_voxels == 2 * size * size
    assert condition.conflict_rate == size / (2 * size * size)
    assert int(condition.mask.sum()) == 2 * size * size - size
    assert int(condition.axis_masks[:, 0].sum()) == size * size
    assert int(condition.axis_masks[:, 1].sum()) == size * size
    assert int(condition.axis_masks[:, 2].sum()) == 0
    assert torch.all(condition.target[:, 1, 1] == 0)


def test_partial_anchor_is_centered_without_resizing() -> None:
    image = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    condition = build_anchors(
        (PlaneAnchor(image=image, axis=0, index=1),),
        batch_size=1,
        num_phases=3,
        volume_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert condition is not None
    assert condition.source_voxels == 4
    assert int(condition.mask.sum()) == 4
    assert torch.equal(condition.target[0, 1, 1:3, 1:3], image)
    assert torch.equal(
        condition.mask[0, 0, 1, 1:3, 1:3],
        torch.ones(2, 2, dtype=torch.bool),
    )


def test_partial_anchor_accepts_an_explicit_position() -> None:
    image = torch.ones(2, 3, dtype=torch.long)
    condition = build_anchors(
        (PlaneAnchor(image=image, axis=2, index=3, position=(2, 1)),),
        batch_size=1,
        num_phases=3,
        volume_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert condition is not None
    assert torch.equal(condition.target[0, 2:4, 1:4, 3], image)
    assert int(condition.mask.sum()) == 6


def test_partial_anchor_preserves_batch_and_axis_coordinates() -> None:
    images = torch.arange(12).reshape(2, 2, 3).remainder(3).long()
    condition = build_anchors(
        (PlaneAnchor(image=images, axis=1, index=4, position=(1, 0)),),
        batch_size=2,
        num_phases=3,
        volume_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert condition is not None
    assert condition.source_voxels == 12
    assert torch.equal(condition.target[:, 1:3, 4, 0:3], images)
    assert int(condition.mask.sum()) == 12


def test_anchor_image_must_fit_generation_size_and_phase_count() -> None:
    with pytest.raises(ValueError, match="fit inside"):
        build_anchors(
            (
                PlaneAnchor(
                    image=torch.zeros(5, 3, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=3,
            volume_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="outside num_phases"):
        build_anchors(
            (
                PlaneAnchor(
                    image=torch.full((4, 4), 3, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=3,
            volume_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_anchor_image_rejects_fractional_phase_labels() -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        build_anchors(
            (
                PlaneAnchor(
                    image=torch.tensor([[0.2, 1.8]], dtype=torch.float32),
                    axis=0,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=2,
            volume_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
