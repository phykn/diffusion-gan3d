from itertools import pairwise
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor, build_anchors
from src.model.critic import ConnectivityCritic2D
from src.train.connect import Connectivity, TripletBatch, normal_transition_loss


def test_straight_through_anchor_is_hard_and_only_surroundings_receive_gradients() -> (
    None
):
    connect = _connectivity(
        num_phases=3,
        patch_size=5,
        replay_triplets_per_axis=1,
        max_triplets_per_step=2,
    )
    connect.record_unconditional(_constant_prediction(0, 3, 5))
    seed = PlaneAnchor(
        torch.ones(5, 5, dtype=torch.uint8),
        axis=0,
        index=2,
    )
    condition = _condition((seed,), num_phases=3, volume_size=5)
    prediction = torch.randn(1, 3, 5, 5, 5, requires_grad=True)

    real, fake = connect.match_anchor(prediction, condition)

    assert len(fake) == len(real) == 1
    assert set(fake.values.detach().unique().tolist()) == {-1.0, 1.0}
    expected = torch.full((3, 5, 5), -1.0)
    expected[1] = 1.0
    assert torch.equal(fake.values[0, 1], expected)
    assert real.values.requires_grad is False

    fake.values.sum().backward()

    assert prediction.grad is not None
    assert not bool(prediction.grad[:, :, 2].any())
    assert bool(prediction.grad[:, :, 1].any())
    assert bool(prediction.grad[:, :, 3].any())


def test_endpoint_anchors_use_only_real_consecutive_slices() -> None:
    size = 5
    labels = torch.arange(size).view(1, size, 1, 1).expand(1, size, size, size)
    prediction = _prediction_from_labels(labels, num_phases=size)
    connect = _connectivity(
        num_phases=size,
        patch_size=size,
        replay_triplets_per_axis=1,
        max_triplets_per_step=4,
    )
    connect.record_unconditional(prediction)
    seeds = (
        PlaneAnchor(labels[0, 0].to(torch.uint8), axis=0, index=0),
        PlaneAnchor(labels[0, -1].to(torch.uint8), axis=0, index=size - 1),
    )
    condition = _condition(seeds, num_phases=size, volume_size=size)

    real, fake = connect.match_anchor(prediction, condition)
    hard = fake.values.argmax(dim=2)

    assert fake.axes.tolist() == [0, 0]
    assert fake.center_slots.tolist() == [0, 2]
    assert real.center_slots.tolist() == [1, 1]
    assert torch.equal(hard[0, :, 0, 0], torch.tensor((0, 1, 2)))
    assert torch.equal(hard[1, :, 0, 0], torch.tensor((2, 3, 4)))


def test_online_reference_replay_is_cpu_float16_axis_matched_fifo() -> None:
    connect = _connectivity(
        num_phases=2,
        patch_size=4,
        replay_triplets_per_axis=1,
        replay_capacity_per_axis=2,
    )
    connect.record_unconditional(_constant_prediction(0, 2, 4))
    source = _constant_prediction(1, 2, 4, requires_grad=True)
    connect.record_unconditional(source)
    connect.record_unconditional(source)
    source.data.fill_(-9.0)

    assert connect.replay_size == 6
    for axis in (0, 1, 2):
        entries = connect._replay._items[axis]
        assert len(entries) == 2
        for entry in entries:
            assert entry.values.device.type == "cpu"
            assert entry.values.dtype == torch.float16
            assert entry.values.grad_fn is None
            assert bool((entry.values[:, 1] == 1.0).all())

    seed = PlaneAnchor(
        torch.ones(4, 4, dtype=torch.uint8),
        axis=2,
        index=1,
    )
    condition = _condition((seed,), num_phases=2, volume_size=4)
    real, fake = connect.match_anchor(
        _constant_prediction(1, 2, 4),
        condition,
    )

    assert fake.axes.tolist() == [2]
    assert bool((real.values[:, :, 1] == 1.0).all())


def test_unconditional_replay_categorizes_only_sampled_triplets() -> None:
    connect = _connectivity(
        num_phases=2,
        patch_size=4,
        replay_triplets_per_axis=1,
    )
    prediction = torch.randn(1, 2, 8, 8, 8)

    with patch.object(
        connect,
        "_hard_labels",
        side_effect=AssertionError("full-volume conversion must not run"),
    ):
        connect.record_unconditional(prediction)

    assert connect.replay_size == 3


def test_online_reference_matching_never_crosses_axes() -> None:
    size = 5
    labels = torch.arange(size).remainder(3).view(1, size, 1, 1)
    labels = labels.expand(1, size, size, size)
    prediction = _prediction_from_labels(labels, num_phases=3)
    connect = _connectivity(
        num_phases=3,
        patch_size=size,
        replay_triplets_per_axis=1,
        replay_capacity_per_axis=1,
    )
    connect.record_unconditional(prediction)
    seed = PlaneAnchor(labels[0, 2].to(torch.uint8), axis=0, index=2)
    condition = _condition((seed,), num_phases=3, volume_size=size)

    real, fake = connect.match_anchor(prediction, condition)
    phases = real.values.argmax(dim=2)[0, :, 0, 0]

    assert fake.axes.tolist() == [0]
    assert phases.unique().numel() == 3
    assert torch.equal((phases[1:] - phases[:-1]).remainder(3), torch.ones(2))


def test_teacher_storage_is_cpu_uint8_detached_and_hard_overlays_real_seed() -> None:
    connect = _connectivity(num_phases=2, patch_size=4)
    prediction = _constant_prediction(0, 2, 4, requires_grad=True)
    seed_image = torch.ones(4, 4, dtype=torch.uint8)
    seed = PlaneAnchor(seed_image, axis=0, index=0)
    condition = _condition((seed,), num_phases=2, volume_size=4)

    connect.record_seeded(prediction, condition, (seed,))
    prediction.data.fill_(-7.0)
    seed_image.zero_()

    assert connect.teacher_count == 1
    entry = connect._teachers._items[0]
    assert entry.labels.device.type == "cpu"
    assert entry.labels.dtype == torch.uint8
    assert entry.labels.is_contiguous()
    assert entry.labels.grad_fn is None
    assert bool((entry.labels[0] == 1).all())
    assert bool((entry.labels[1:] == 0).all())
    assert bool((entry.seed.image == 1).all())
    assert entry.seed.image.device.type == "cpu"
    assert entry.seed.image.dtype == torch.uint8
    assert torch.allclose(entry.target_vf, torch.tensor((0.75, 0.25)))


def test_teacher_bank_uses_one_global_fifo_byte_budget_across_volume_sizes() -> None:
    connect = _connectivity(
        num_phases=3,
        patch_size=3,
        teacher_bank_bytes=85,
    )
    _record_constant_teacher(connect, phase=0, volume_size=3)
    assert connect.teacher_count_for(3) == 1

    _record_constant_teacher(connect, phase=2, volume_size=4)

    assert connect.teacher_storage_bytes == 85
    assert connect.teacher_count == 1
    assert connect.teacher_count_for(3) == 0
    assert connect.teacher_count_for(4) == 1
    assert bool((connect._teachers._items[0].labels[1:] == 2).all())


def test_record_seeded_splits_prediction_batch_into_independent_teachers() -> None:
    connect = _connectivity(num_phases=3, patch_size=3)
    labels = torch.stack(
        (
            torch.zeros(4, 4, 4, dtype=torch.long),
            torch.full((4, 4, 4), 2, dtype=torch.long),
        )
    )
    prediction = _prediction_from_labels(labels, num_phases=3)
    seed_images = torch.stack(
        (
            torch.ones(3, 3, dtype=torch.uint8),
            torch.ones(3, 3, dtype=torch.uint8),
        )
    )
    batched_seed = PlaneAnchor(
        seed_images,
        axis=0,
        index=0,
        position=(0, 0),
    )
    condition = _condition(
        (batched_seed,),
        num_phases=3,
        volume_size=4,
        batch_size=2,
    )
    seeds = tuple(
        PlaneAnchor(image, axis=0, index=0, position=(0, 0)) for image in seed_images
    )

    connect.record_seeded(prediction, condition, seeds)

    assert connect.teacher_count == 2
    first, second = connect._teachers._items
    assert first.labels.shape == second.labels.shape == (4, 4, 4)
    assert bool((first.labels[1:] == 0).all())
    assert bool((second.labels[1:] == 2).all())
    assert first.seed.image.ndim == second.seed.image.ndim == 2


def test_teacher_sampling_supports_more_than_four_mixed_axis_strict_anchors() -> None:
    connect, labels, teacher = _mixed_teacher(max_triplets_per_step=4)
    condition = teacher.condition

    assert condition.planes == 8
    assert condition.conflicts == 0
    active_axes = condition.axis_masks.flatten(2).any(dim=(0, 2))
    assert active_axes.tolist() == [True, True, True]
    selected = condition.mask[0, 0]
    assert torch.equal(condition.target[0][selected], labels[0][selected])

    for axis in (0, 1, 2):
        indices = (
            condition.axis_masks[0, axis]
            .movedim(axis, 0)
            .flatten(1)
            .any(dim=1)
            .nonzero()
            .flatten()
            .tolist()
        )
        assert all(right - left >= 2 for left, right in pairwise(indices))

    counts = torch.bincount(labels.flatten(), minlength=3).to(torch.float32)
    assert torch.allclose(teacher.target_vf[0], counts / counts.sum())
    assert connect.teacher_count_for(8) == 1


def test_connectivity_triplet_work_stays_bounded_for_many_mixed_anchors() -> None:
    connect, labels, teacher = _mixed_teacher(max_triplets_per_step=2)
    prediction = _prediction_from_labels(labels, num_phases=3, requires_grad=True)
    connect.record_unconditional(prediction.detach())

    real, fake = connect.match_anchor(prediction, teacher.condition)

    assert teacher.condition.planes > 4
    assert len(fake) == len(real) == 2
    assert fake.values.requires_grad
    assert set(fake.axes.tolist()).issubset({0, 1, 2})


def test_normal_transition_loss_is_zero_for_matching_triplets() -> None:
    labels = torch.tensor([[[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]]])
    values = (
        F.one_hot(labels, num_classes=2)
        .movedim(-1, 2)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )
    batch = TripletBatch(
        values=values,
        axes=torch.tensor((2,)),
        center_slots=torch.tensor((1,)),
    )

    loss = normal_transition_loss(batch, batch)

    assert float(loss) == 0.0


def test_normal_transition_loss_measures_neighbor_tv_and_backpropagates() -> None:
    real_labels = torch.zeros(1, 3, 2, 2, dtype=torch.long)
    fake_labels = real_labels.clone()
    fake_labels[:, (0, 2)] = 1
    real_values = (
        F.one_hot(real_labels, num_classes=2)
        .movedim(-1, 2)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )
    fake_values = (
        F.one_hot(fake_labels, num_classes=2)
        .movedim(-1, 2)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
        .requires_grad_()
    )
    axes = torch.tensor((0,))
    center_slots = torch.tensor((1,))
    real = TripletBatch(
        values=real_values,
        axes=axes,
        center_slots=center_slots,
    )
    fake = TripletBatch(
        values=fake_values,
        axes=axes,
        center_slots=center_slots,
    )

    loss = normal_transition_loss(real, fake)
    reversed_loss = normal_transition_loss(
        TripletBatch(
            values=real_values.flip(1),
            axes=axes,
            center_slots=center_slots,
        ),
        TripletBatch(
            values=fake_values.flip(1),
            axes=axes,
            center_slots=center_slots,
        ),
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor(1.0))
    assert torch.equal(loss, reversed_loss)
    assert fake_values.grad is not None
    assert bool(torch.isfinite(fake_values.grad).all())
    assert float(fake_values.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("center_slot", "fake_labels", "changed_neighbor_labels"),
    (
        (0, (0, 2, 1), (0, 1, 1)),
        (2, (2, 1, 0), (2, 2, 0)),
    ),
)
def test_normal_transition_loss_aligns_replay_and_endpoint_sides(
    center_slot: int,
    fake_labels: tuple[int, int, int],
    changed_neighbor_labels: tuple[int, int, int],
) -> None:
    real_labels = torch.tensor((1, 0, 2)).reshape(1, 3, 1, 1)

    def triplets(labels: tuple[int, int, int], center: int) -> TripletBatch:
        label_tensor = torch.tensor(labels).reshape(1, 3, 1, 1)
        values = (
            F.one_hot(label_tensor, num_classes=3)
            .movedim(-1, 2)
            .to(torch.float32)
            .mul(2.0)
            .sub(1.0)
        )
        return TripletBatch(
            values=values,
            axes=torch.tensor((0,)),
            center_slots=torch.tensor((center,)),
        )

    real = triplets(tuple(real_labels.flatten().tolist()), center=1)

    assert (
        float(normal_transition_loss(real, triplets(fake_labels, center_slot))) == 0.0
    )
    assert (
        float(
            normal_transition_loss(
                real,
                triplets(changed_neighbor_labels, center_slot),
            )
        )
        > 0.0
    )


def test_triplet_batch_index_select_preserves_center_slots() -> None:
    batch = TripletBatch(
        values=torch.arange(18).reshape(3, 3, 2, 1, 1).to(torch.float32),
        axes=torch.tensor((0, 1, 2)),
        center_slots=torch.tensor((0, 1, 2)),
    )

    selected = batch.index_select(torch.tensor((2, 0)))

    assert selected.axes.tolist() == [2, 0]
    assert selected.center_slots.tolist() == [2, 0]
    assert torch.equal(selected.values, batch.values[(2, 0), ...])


@pytest.mark.parametrize(("volume_size", "maximum"), ((128, 6), (160, 12)))
def test_multi_anchor_count_starts_at_two_and_scales_to_max_density(
    volume_size: int,
    maximum: int,
) -> None:
    connect = _connectivity(patch_size=128, max_density=0.05)

    with patch(
        "src.train.connect.torch.randint",
        return_value=torch.tensor(maximum),
    ) as randint:
        count = connect._sample_plane_count(volume_size, generator=None)

    randint.assert_called_once_with(2, maximum + 1, (), generator=None)
    assert count == maximum


@pytest.mark.parametrize("num_phases", (2, 3, 5))
def test_connectivity_critic_is_multiphase_and_exactly_reversal_invariant(
    num_phases: int,
) -> None:
    critic = ConnectivityCritic2D(
        num_phases=num_phases,
        channels=(4, 8),
        embedding_channels=8,
        reversal_invariant=True,
        gradient_checkpointing=False,
    )
    triplets = torch.randn(3, 3, num_phases, 15, 17, requires_grad=True)
    axes = torch.tensor((0, 1, 2))

    forward = critic(triplets, axes)
    reverse = critic(triplets.flip(1), axes)

    assert forward.logits_global.shape == (3,)
    assert forward.logits_local.shape[0] == 3
    assert torch.equal(forward.logits_global, reverse.logits_global)
    assert torch.equal(forward.logits_local, reverse.logits_local)
    (forward.logits_global.sum() + forward.logits_local.sum()).backward()
    assert triplets.grad is not None
    assert bool(torch.isfinite(triplets.grad).all())


def test_connectivity_critic_rejects_invalid_axes() -> None:
    critic = ConnectivityCritic2D(
        num_phases=2,
        channels=(4, 8),
        embedding_channels=8,
    )
    with pytest.raises(ValueError, match="only 0, 1, or 2"):
        critic(torch.randn(1, 3, 2, 8, 8), torch.tensor((3,)))


def _connectivity(
    *,
    num_phases: int = 3,
    patch_size: int = 4,
    replay_triplets_per_axis: int = 2,
    replay_capacity_per_axis: int = 4,
    max_triplets_per_step: int = 4,
    teacher_bank_bytes: int = 1_000_000,
    teacher_min_entries: int = 1,
    max_density: float = 0.25,
    min_spacing: int = 2,
    mixed_axis_probability: float = 1.0,
) -> Connectivity:
    return Connectivity(
        num_phases=num_phases,
        patch_size=patch_size,
        replay_triplets_per_axis=replay_triplets_per_axis,
        replay_capacity_per_axis=replay_capacity_per_axis,
        max_triplets_per_step=max_triplets_per_step,
        teacher_bank_bytes=teacher_bank_bytes,
        teacher_min_entries=teacher_min_entries,
        max_density=max_density,
        min_spacing=min_spacing,
        mixed_axis_probability=mixed_axis_probability,
    )


def _condition(
    seeds: tuple[PlaneAnchor, ...],
    *,
    num_phases: int,
    volume_size: int,
    batch_size: int = 1,
):
    condition = build_anchors(
        seeds,
        batch_size=batch_size,
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
    requires_grad: bool = False,
) -> torch.Tensor:
    prediction = (
        F.one_hot(labels.to(torch.long), num_classes=num_phases)
        .movedim(-1, 1)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )
    return prediction.requires_grad_(requires_grad)


def _constant_prediction(
    phase: int,
    num_phases: int,
    size: int,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    labels = torch.full((1, size, size, size), phase, dtype=torch.long)
    return _prediction_from_labels(
        labels,
        num_phases=num_phases,
        requires_grad=requires_grad,
    )


def _record_constant_teacher(
    connect: Connectivity,
    *,
    phase: int,
    volume_size: int,
) -> None:
    prediction = _constant_prediction(phase, connect.num_phases, volume_size)
    seed_image = torch.full(
        (connect.patch_size, connect.patch_size),
        phase,
        dtype=torch.uint8,
    )
    seed = PlaneAnchor(seed_image, axis=0, index=0, position=(0, 0))
    condition = _condition(
        (seed,),
        num_phases=connect.num_phases,
        volume_size=volume_size,
    )
    connect.record_seeded(prediction, condition, (seed,))


def _mixed_teacher(*, max_triplets_per_step: int):
    size = 8
    coordinates = torch.stack(torch.meshgrid(*(torch.arange(size),) * 3, indexing="ij"))
    labels = coordinates.sum(dim=0).remainder(3).unsqueeze(0)
    prediction = _prediction_from_labels(labels, num_phases=3)
    connect = _connectivity(
        num_phases=3,
        patch_size=4,
        max_triplets_per_step=max_triplets_per_step,
        max_density=0.25,
        min_spacing=2,
        mixed_axis_probability=1.0,
    )
    seed = PlaneAnchor(
        labels[0, 0, :4, :4].to(torch.uint8),
        axis=0,
        index=0,
        position=(0, 0),
    )
    condition = _condition((seed,), num_phases=3, volume_size=size)
    connect.record_seeded(prediction, condition, (seed,))
    with patch.object(connect, "_sample_plane_count", return_value=8):
        teacher = connect.sample_teacher(
            volume_size=size,
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(7),
        )
    assert teacher is not None
    return connect, labels, teacher
