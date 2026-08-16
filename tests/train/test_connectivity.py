from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor, build_anchors
from src.model.critic import ConnectivityCritic2D, connectivity_images
from src.train.connect import Connectivity, TripletBatch, normal_transition_loss


def test_prior_bank_stores_complete_cpu_uint8_volumes_and_freezes() -> None:
    size = 4
    connect = _connectivity(num_phases=2, patch_size=size, volumes=2)
    first = torch.zeros(1, size, size, size, dtype=torch.long)
    second = torch.ones_like(first)

    _record_prior(connect, _prediction_from_labels(first, num_phases=2), 0)
    assert not connect.prior_ready
    _record_prior(connect, _prediction_from_labels(second, num_phases=2), 0)
    assert connect.prior_ready
    _record_prior(connect, _prediction_from_labels(first, num_phases=2), 0)

    assert connect.prior_count == 2
    assert connect.prior_storage_bytes == 2 * (size**3 + size**2)
    entries = connect.prior._banks[0].items
    assert all(item.labels.device.type == "cpu" for item in entries)
    assert all(item.labels.dtype == torch.uint8 for item in entries)
    assert torch.equal(
        entries[1].labels,
        second[0].to(torch.uint8),
    )


def test_prior_uses_domain_axis_first_and_provider_fallback_for_missing_axis() -> None:
    size = 4
    connect = _connectivity(
        num_phases=2,
        num_domains=2,
        patch_size=size,
        volumes=1,
        owned_axes={0: (0,), 1: (1,)},
    )
    zeros = torch.zeros(1, size, size, size, dtype=torch.long)
    ones = torch.ones_like(zeros)
    _record_prior(connect, _prediction_from_labels(zeros, num_phases=2), 0)
    _record_prior(connect, _prediction_from_labels(ones, num_phases=2), 1)

    own = connect._prior_volumes(0, 0)
    borrowed = connect._prior_volumes(0, 1)

    assert len(own) == len(borrowed) == 1
    assert int(own[0].max()) == 0
    assert int(borrowed[0].min()) == 1


def test_refresh_pushes_one_volume_and_evicts_the_oldest() -> None:
    size = 4
    connect = _connectivity(
        num_phases=3,
        patch_size=size,
        volumes=2,
    )
    zeros = torch.zeros(1, size, size, size, dtype=torch.long)
    ones = torch.ones_like(zeros)
    twos = torch.full_like(zeros, 2)
    _record_prior(connect, _prediction_from_labels(zeros, num_phases=3), 0)
    _record_prior(connect, _prediction_from_labels(ones, num_phases=3), 0)
    storage_bytes = connect.prior_storage_bytes

    _refresh_prior(connect, _prediction_from_labels(twos, num_phases=3), 0)

    volumes = connect._prior_volumes(0, 0)
    assert len(volumes) == 2
    assert int(volumes[0].min()) == 1
    assert int(volumes[1].min()) == 2
    assert connect.prior_count == 2
    assert connect.prior_storage_bytes == storage_bytes
    assert connect.prior_updates == 1


def test_straight_through_triplets_backpropagate_to_the_prediction() -> None:
    size = 5
    connect = _connectivity(num_phases=3, patch_size=size, volumes=1)
    prior = torch.zeros(1, size, size, size, dtype=torch.long)
    _record_prior(connect, _prediction_from_labels(prior, num_phases=3), 0)
    prediction = torch.randn(1, 3, size, size, size, requires_grad=True)
    seed = PlaneAnchor(
        prediction.detach()[0].argmax(dim=0)[2].to(torch.uint8),
        axis=0,
        index=2,
    )

    _, fake = connect.match_anchor(
        prediction,
        _condition((seed,), num_phases=3, volume_size=size),
        0,
    )
    fake.values.sum().backward()

    assert len(fake) == 5
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0


def test_change_images_ignore_confidence_but_keep_student_gradients() -> None:
    connect = _connectivity(num_phases=2, patch_size=4)
    labels = torch.tensor((0, 1, 1)).view(1, 1, 3, 1, 1).expand(1, 1, 3, 4, 4)
    one_hot = F.one_hot(labels[:, 0], num_classes=2).movedim(-1, 1).float()
    low_confidence = (one_hot * 0.6 + (1.0 - one_hot) * 0.4).requires_grad_()
    high_confidence = one_hot * 10.0 - (1.0 - one_hot) * 10.0

    low_images = connectivity_images(
        connect._straight_through(low_confidence).movedim(1, 2)
    )
    high_images = connectivity_images(
        connect._straight_through(high_confidence).movedim(1, 2)
    )
    loss = low_images.square().sum()
    loss.backward()

    assert torch.equal(low_images, high_images)
    assert low_confidence.grad is not None
    assert float(low_confidence.grad.abs().sum()) > 0.0


def test_prior_match_checks_every_ready_volume_and_selects_nearest_fraction() -> None:
    size = 4
    connect = _connectivity(num_phases=2, patch_size=size, volumes=1)
    target = _triplet_batch(
        torch.ones(1, 3, size, size, dtype=torch.long),
        num_phases=2,
    )
    volumes = (
        torch.zeros(size, size, size, dtype=torch.uint8),
        torch.ones(size, size, size, dtype=torch.uint8),
    )

    with (
        patch.object(connect, "_prior_volumes", return_value=volumes),
        patch.object(
            connect,
            "_sample_prior_triplet",
            wraps=connect._sample_prior_triplet,
        ) as sampled,
    ):
        real, matched = connect._sample_prior_matches(
            target,
            domain=0,
            generator=torch.Generator().manual_seed(1),
        )

    assert sampled.call_count == len(volumes)
    assert matched.tolist() == [0]
    assert bool((real.values.argmax(dim=2) == 1).all())


def test_endpoint_anchors_use_only_real_consecutive_slices() -> None:
    size = 5
    labels = torch.arange(size).view(1, size, 1, 1).expand(1, size, size, size)
    prediction = _prediction_from_labels(labels, num_phases=size)
    connect = _connectivity(num_phases=size, patch_size=size)
    seeds = (
        PlaneAnchor(labels[0, 0].to(torch.uint8), axis=0, index=0),
        PlaneAnchor(labels[0, 4].to(torch.uint8), axis=0, index=4),
    )

    fake = connect._sample_anchor_triplets(
        connect._straight_through(prediction),
        _condition(seeds, num_phases=size, volume_size=size),
    )
    hard = fake.values.argmax(dim=2)

    assert fake.axes.tolist() == [0, 0]
    assert fake.center_slots.tolist() == [0, 2]
    assert torch.equal(hard[0, :, 0, 0], torch.tensor((0, 1, 2)))
    assert torch.equal(hard[1, :, 0, 0], torch.tensor((2, 3, 4)))


def test_anchor_match_adds_two_general_windows_from_every_axis() -> None:
    size = 9
    labels = torch.arange(size).view(1, size, 1, 1).expand(1, size, size, size)
    prediction = _prediction_from_labels(labels, num_phases=size)
    connect = _connectivity(num_phases=size, patch_size=size, volumes=1)
    _record_prior(connect, prediction, 0)
    condition = _condition(
        (PlaneAnchor(labels[0, 4].to(torch.uint8), axis=0, index=4),),
        num_phases=size,
        volume_size=size,
    )

    with patch(
        "src.train.connect.torch.randperm",
        return_value=torch.tensor((0, 1, 2, 3)),
    ):
        _, fake = connect.match_anchor(prediction, condition, 0)

    assert len(fake) == 7
    assert fake.axes.tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert fake.center_slots.tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert fake.anchor_flags.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_multi_anchor_match_keeps_total_budget_and_prioritizes_anchor_windows() -> None:
    size = 9
    coordinates = torch.meshgrid(
        *(torch.arange(size) for _ in range(3)),
        indexing="ij",
    )
    labels = sum(coordinates).remainder(3).unsqueeze(0)
    prediction = _prediction_from_labels(labels, num_phases=3)
    connect = _connectivity(num_phases=3, patch_size=size, volumes=1)
    _record_prior(connect, prediction, 0)
    anchors = tuple(
        PlaneAnchor(labels[0].select(axis, index).to(torch.uint8), axis, index)
        for axis, index in ((0, 2), (1, 4), (2, 6))
    )

    _, fake = connect.match_anchor(
        prediction,
        _condition(anchors, num_phases=3, volume_size=size),
        0,
    )

    assert len(fake) == 7
    assert int(fake.anchor_flags.sum()) == 3
    assert int((~fake.anchor_flags).sum()) == 4


def test_many_anchors_keep_general_axis_coverage_within_fixed_budget() -> None:
    size = 15
    coordinates = torch.meshgrid(
        *(torch.arange(size) for _ in range(3)),
        indexing="ij",
    )
    labels = sum(coordinates).remainder(3).unsqueeze(0)
    prediction = _prediction_from_labels(labels, num_phases=3)
    connect = _connectivity(num_phases=3, patch_size=size, volumes=1)
    _record_prior(connect, prediction, 0)
    anchors = tuple(
        PlaneAnchor(labels[0].select(axis, index).to(torch.uint8), axis, index)
        for axis in range(3)
        for index in (2, 7, 12)
    )

    _, fake = connect.match_anchor(
        prediction,
        _condition(anchors, num_phases=3, volume_size=size),
        0,
    )

    assert len(fake) == 7
    assert int(fake.anchor_flags.sum()) == 4
    assert set(fake.axes[~fake.anchor_flags].tolist()) == {0, 1, 2}


def test_connectivity_images_are_phase_changes_and_discrete_bend() -> None:
    phases = torch.tensor((0.0, 0.25, 1.0)).reshape(1, 3, 1, 1, 1)
    triplets = phases.mul(2.0).sub(1.0)

    images = connectivity_images(triplets)

    assert torch.allclose(images[:, 0], torch.tensor(0.25))
    assert torch.allclose(images[:, 1], torch.tensor(0.75))
    assert torch.allclose(images[:, 2], torch.tensor(0.25))
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0


def test_connectivity_images_remove_constant_slice_appearance() -> None:
    first = torch.full((1, 3, 2, 4, 4), -1.0)
    second = torch.full_like(first, 1.0)

    assert not torch.equal(first, second)
    assert not bool(connectivity_images(first).any())
    assert not bool(connectivity_images(second).any())


def test_normal_transition_loss_is_zero_for_matching_triplets() -> None:
    labels = torch.tensor([[[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]]])
    batch = _triplet_batch(labels, num_phases=2)

    assert float(normal_transition_loss(batch, batch)) == 0.0


def test_normal_transition_loss_measures_neighbor_tv_and_backpropagates() -> None:
    real_labels = torch.zeros(1, 3, 2, 2, dtype=torch.long)
    fake_labels = real_labels.clone()
    fake_labels[:, (0, 2)] = 1
    real = _triplet_batch(real_labels, num_phases=2)
    fake_values = _triplet_values(fake_labels, num_phases=2).requires_grad_()
    fake = TripletBatch(fake_values, real.axes, real.center_slots, real.anchor_flags)

    loss = normal_transition_loss(real, fake)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(1.0))
    assert fake_values.grad is not None
    assert bool(torch.isfinite(fake_values.grad).all())


def test_normal_transition_balances_anchor_and_general_groups() -> None:
    real_labels = torch.zeros(4, 3, 2, 2, dtype=torch.long)
    fake_labels = real_labels.clone()
    fake_labels[0, (0, 2)] = 1
    flags = torch.tensor((True, False, False, False))
    real = _triplet_batch(real_labels, num_phases=2, anchor_flags=flags)
    fake = _triplet_batch(fake_labels, num_phases=2, anchor_flags=flags)

    loss = normal_transition_loss(real, fake)

    assert torch.isclose(loss, torch.tensor(0.5))


@pytest.mark.parametrize("num_phases", (2, 3, 5))
def test_connectivity_critic_is_multiphase_and_exactly_reversal_invariant(
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
    domains = torch.zeros(3, dtype=torch.long)

    forward = critic(triplets, axes, domains)
    reverse = critic(triplets.flip(1), axes, domains)

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
            torch.zeros(1, dtype=torch.long),
        )


def _connectivity(
    *,
    num_phases: int = 3,
    num_domains: int = 1,
    patch_size: int = 4,
    volumes: int = 1,
    owned_axes: dict[int, tuple[int, ...]] | None = None,
) -> Connectivity:
    if owned_axes is None:
        owned_axes = {domain: (0, 1, 2) for domain in range(num_domains)}
    return Connectivity(
        num_phases=num_phases,
        num_domains=num_domains,
        patch_size=patch_size,
        bank_size=volumes,
        owned_axes=owned_axes,
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


def _record_prior(
    connect: Connectivity,
    prediction: torch.Tensor,
    domain: int,
) -> None:
    connect.record_prior(
        prediction,
        _observed_condition(prediction, connect.num_phases),
        domain,
    )


def _refresh_prior(
    connect: Connectivity,
    prediction: torch.Tensor,
    domain: int,
) -> None:
    connect.refresh_prior(
        prediction,
        _observed_condition(prediction, connect.num_phases),
        domain,
    )


def _observed_condition(prediction: torch.Tensor, num_phases: int):
    labels = prediction.detach().argmax(dim=1)
    index = labels.shape[1] // 2
    return _condition(
        (PlaneAnchor(labels[0, index].to(torch.uint8), axis=0, index=index),),
        num_phases=num_phases,
        volume_size=labels.shape[1],
    )


def _triplet_values(labels: torch.Tensor, *, num_phases: int) -> torch.Tensor:
    return (
        F.one_hot(labels.to(torch.long), num_classes=num_phases)
        .movedim(-1, 2)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )


def _triplet_batch(
    labels: torch.Tensor,
    *,
    num_phases: int,
    anchor_flags: torch.Tensor | None = None,
) -> TripletBatch:
    if anchor_flags is None:
        anchor_flags = torch.zeros(labels.shape[0], dtype=torch.bool)
    return TripletBatch(
        values=_triplet_values(labels, num_phases=num_phases),
        axes=torch.zeros(labels.shape[0], dtype=torch.long),
        center_slots=torch.ones(labels.shape[0], dtype=torch.long),
        anchor_flags=anchor_flags,
    )
