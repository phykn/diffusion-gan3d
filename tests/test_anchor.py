import pytest
import torch

from src.anchor import PlaneAnchor, assemble_anchors


def test_plane_anchor_builds_dense_condition_for_every_axis() -> None:
    size = 4
    labels = torch.arange(size * size).reshape(size, size).remainder(3).long()

    for axis in (0, 1, 2):
        condition = assemble_anchors(
            (PlaneAnchor(labels=labels, axis=axis, index=2),),
            batch_size=1,
            num_phases=3,
            volume_size=size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert condition is not None
        assert condition.image.shape == (1, 3, size, size, size)
        assert condition.mask.shape == (1, 1, size, size, size)
        assert int(condition.mask.sum()) == size * size
        generated = condition.image.argmax(dim=1).select(axis + 1, 2)
        assert torch.equal(generated[0], labels)


def test_cross_axis_anchor_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting intersections"):
        assemble_anchors(
            (
                PlaneAnchor(
                    labels=torch.zeros(4, 4, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
                PlaneAnchor(
                    labels=torch.ones(4, 4, dtype=torch.long),
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


def test_anchor_labels_must_match_generation_size_and_phase_count() -> None:
    with pytest.raises(ValueError, match="plane size"):
        assemble_anchors(
            (
                PlaneAnchor(
                    labels=torch.zeros(3, 3, dtype=torch.long),
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
        assemble_anchors(
            (
                PlaneAnchor(
                    labels=torch.full((4, 4), 3, dtype=torch.long),
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
