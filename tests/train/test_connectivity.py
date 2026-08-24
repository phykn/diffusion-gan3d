import pytest
import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor, build_anchors
from src.model.critic import ConnectivityCritic2D
from src.train.connect import Connectivity, TripletBatch, normal_transition_loss


def test_anchor_triplets_cover_all_axes_and_intersect_the_anchor() -> None:
    size = 7
    labels = torch.arange(size).view(1, size, 1, 1).expand(1, size, size, size)
    prediction = _prediction_from_labels(labels, num_phases=size)
    connect = _connectivity(num_phases=size, patch_size=size, max_gap=2)
    condition = _condition(
        (PlaneAnchor(labels[0, 3].to(torch.uint8), axis=0, index=3),),
        num_phases=size,
        volume_size=size,
    )

    located = connect._sample_anchor_triplets(
        connect._straight_through(prediction),
        condition,
    )

    assert located.triplets.axes.tolist() == [0, 1, 2]
    for batch, axis, index in located.locations:
        assert bool(condition.mask[batch, 0].select(axis, index).any())


def test_anchor_match_uses_the_same_triplet_metadata_for_real_and_fake() -> None:
    size = 7
    labels = torch.arange(size).view(1, size, 1, 1).expand(1, size, size, size)
    conditioned = _prediction_from_labels(labels, num_phases=size).requires_grad_()
    reference = _prediction_from_labels(labels.remainder(size - 1), num_phases=size)
    condition = _condition(
        (PlaneAnchor(labels[0, 3].to(torch.uint8), axis=0, index=3),),
        num_phases=size,
        volume_size=size,
    )
    connect = _connectivity(num_phases=size, patch_size=size, max_gap=2)

    real, fake = connect.match_anchor(conditioned, reference, condition)

    assert len(real) == len(fake) == 3
    assert torch.equal(real.axes, fake.axes)
    assert torch.equal(real.gaps, fake.gaps)
    assert torch.equal(real.center_slots, fake.center_slots)
    fake.values.sum().backward()
    assert conditioned.grad is not None
    assert float(conditioned.grad.abs().sum()) > 0.0


def test_endpoint_anchor_uses_an_endpoint_center_slot() -> None:
    size = 5
    labels = torch.arange(size).view(1, size, 1, 1).expand(1, size, size, size)
    prediction = _prediction_from_labels(labels, num_phases=size)
    connect = _connectivity(num_phases=size, patch_size=size)
    condition = _condition(
        (PlaneAnchor(labels[0, 0].to(torch.uint8), axis=0, index=0),),
        num_phases=size,
        volume_size=size,
    )

    located = connect._sample_anchor_triplets(
        connect._straight_through(prediction),
        condition,
    )
    axis_zero = located.triplets.axes == 0

    assert int(axis_zero.sum()) == 1
    assert int(located.triplets.center_slots[axis_zero][0]) == 0


def test_normal_transition_loss_is_zero_for_matching_triplets() -> None:
    labels = torch.tensor(
        [
            [
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ]
        ]
    )
    batch = _triplet_batch(labels, num_phases=2)

    assert float(normal_transition_loss(batch, batch)) == 0.0


def test_normal_transition_loss_measures_neighbor_tv_and_backpropagates() -> None:
    real_labels = torch.zeros(1, 3, 2, 2, dtype=torch.long)
    fake_labels = real_labels.clone()
    fake_labels[:, (0, 2)] = 1
    real = _triplet_batch(real_labels, num_phases=2)
    fake_values = _triplet_values(fake_labels, num_phases=2).requires_grad_()
    fake = TripletBatch(
        values=fake_values,
        axes=real.axes,
        gaps=real.gaps,
        center_slots=real.center_slots,
    )

    loss = normal_transition_loss(real, fake)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(1.0))
    assert fake_values.grad is not None
    assert bool(torch.isfinite(fake_values.grad).all())


def test_connectivity_images_are_phase_changes_and_discrete_bend() -> None:
    phases = torch.tensor((0.0, 0.25, 1.0)).reshape(1, 3, 1, 1, 1)
    triplets = phases.mul(2.0).sub(1.0)

    images = ConnectivityCritic2D.connectivity_images(triplets)

    assert torch.allclose(images[:, 0], torch.tensor(0.25))
    assert torch.allclose(images[:, 1], torch.tensor(0.75))
    assert torch.allclose(images[:, 2], torch.tensor(0.25))
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0


def test_connectivity_images_remove_constant_slice_appearance() -> None:
    first = torch.full((1, 3, 2, 4, 4), -1.0)
    second = torch.full_like(first, 1.0)

    assert not torch.equal(first, second)
    assert not bool(ConnectivityCritic2D.connectivity_images(first).any())
    assert not bool(ConnectivityCritic2D.connectivity_images(second).any())


@pytest.mark.parametrize("num_phases", (2, 3, 5))
def test_connectivity_critic_is_multiphase_and_reversal_invariant(
    num_phases: int,
) -> None:
    critic = ConnectivityCritic2D(
        num_phases=num_phases,
        channels=(4, 8),
        embedding_channels=8,
        num_domains=2,
        gradient_checkpointing=False,
    )
    triplets = torch.randn(3, 3, num_phases, 15, 17, requires_grad=True)
    axes = torch.tensor((0, 1, 2))
    gaps = torch.ones(3, dtype=torch.long)
    domains = torch.zeros(3, dtype=torch.long)

    forward = critic(triplets, axes, gaps, domains)
    reverse = critic(triplets.flip(1), axes, gaps, domains)

    assert torch.equal(forward.logits_global, reverse.logits_global)
    assert torch.equal(forward.logits_local, reverse.logits_local)
    (forward.logits_global.sum() + forward.logits_local.sum()).backward()
    assert triplets.grad is not None


def test_connectivity_critic_rejects_invalid_axes() -> None:
    critic = ConnectivityCritic2D(
        num_phases=2,
        channels=(4, 8),
        embedding_channels=8,
        num_domains=2,
    )
    with pytest.raises(ValueError, match="only 0, 1, or 2"):
        critic(
            torch.randn(1, 3, 2, 8, 8),
            torch.tensor((3,)),
            torch.ones(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )


def _connectivity(
    *,
    num_phases: int = 3,
    patch_size: int = 4,
    max_gap: int = 1,
) -> Connectivity:
    return Connectivity(
        num_phases=num_phases,
        max_gap=max_gap,
    )


def _condition(
    seeds: tuple[PlaneAnchor, ...],
    *,
    num_phases: int,
    volume_size: int,
):
    condition = build_anchors(
        seeds,
        batch_size=1,
        num_phases=num_phases,
        volume_size=volume_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reconcile=False,
    )
    assert condition is not None
    return condition


def _prediction_from_labels(
    labels: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    return (
        F.one_hot(labels.to(torch.long), num_classes=num_phases)
        .movedim(-1, 1)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )


def _triplet_values(labels: torch.Tensor, *, num_phases: int) -> torch.Tensor:
    return (
        F.one_hot(labels.to(torch.long), num_classes=num_phases)
        .movedim(-1, 2)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )


def _triplet_batch(labels: torch.Tensor, *, num_phases: int) -> TripletBatch:
    return TripletBatch(
        values=_triplet_values(labels, num_phases=num_phases),
        axes=torch.zeros(labels.shape[0], dtype=torch.long),
        gaps=torch.ones(labels.shape[0], dtype=torch.long),
        center_slots=torch.ones(labels.shape[0], dtype=torch.long),
    )
