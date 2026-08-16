from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor, build_anchors
from src.train.prior import ConditionalPrior, _log_uniform_count


def test_record_keeps_raw_relation_volume_and_preserves_observed_condition() -> None:
    size = 8
    prior = _prior(size=size, volume_count=1)
    labels = torch.zeros(1, size, size, size, dtype=torch.long)
    image = torch.ones(1, 2, 3, dtype=torch.long)
    observed = _observed(
        image,
        axis=1,
        index=4,
        position=(2, 1),
        size=size,
    )

    prior.record(_prediction(labels), observed, domain=0)

    stored = prior.volumes(0, 1)[0]
    assert stored.device.type == "cpu"
    assert stored.dtype == torch.uint8
    assert not bool(stored.any())
    with patch("src.train.prior._log_uniform_count", return_value=1):
        sampled = prior.sample_condition(
            0,
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert sampled.condition.planes == 1
    assert torch.equal(sampled.observed_mask, observed.mask)
    assert torch.equal(sampled.observed_axis_masks, observed.axis_masks)
    selected = sampled.observed_mask[:, 0]
    assert bool((sampled.condition.target[selected] == 1).all())
    assert sampled.condition.conflicts == 0


def test_sampling_uses_distinct_ready_entries_for_batch_items() -> None:
    size = 8
    prior = _prior(size=size, volume_count=2)
    labels = torch.stack(
        (
            torch.zeros(size, size, size, dtype=torch.long),
            torch.ones(size, size, size, dtype=torch.long),
        )
    )
    images = torch.stack(
        (
            torch.zeros(size, size, dtype=torch.long),
            torch.ones(size, size, dtype=torch.long),
        )
    )
    observed = _observed(images, axis=0, index=3, size=size)
    prior.record(_prediction(labels), observed, domain=0)

    with patch("src.train.prior._log_uniform_count", return_value=1):
        sampled = prior.sample_condition(
            0,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(3),
        )

    roots = sampled.condition.target.select(1, 3).to(torch.float32).mean(dim=(1, 2))
    assert set(roots.tolist()) == {0.0, 1.0}
    assert sampled.condition.image.shape[0] == 2
    assert torch.equal(sampled.condition.mask, sampled.observed_mask)
    assert len(sampled.references) == 2
    assert all(not references for references in sampled.references)


def test_variable_plane_count_keeps_only_the_real_plane_in_observed_mask() -> None:
    size = 8
    prior = _prior(size=size, volume_count=1)
    labels = torch.arange(size**3).reshape(1, size, size, size).remainder(2)
    image = labels[:, 2, 2:6, 1:7]
    observed = _observed(
        image,
        axis=0,
        index=2,
        position=(2, 1),
        size=size,
    )
    prior.record(_prediction(labels), observed, domain=0)

    with patch("src.train.prior._log_uniform_count", side_effect=(1, 4)):
        single = prior.sample_condition(
            0,
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        multiple = prior.sample_condition(
            0,
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert single.condition.planes == 1
    assert multiple.condition.planes == 4
    assert int(multiple.observed_mask.sum()) == image.numel()
    assert torch.equal(
        multiple.observed_axis_masks.any(dim=1, keepdim=True), multiple.observed_mask
    )
    assert not bool((multiple.observed_mask & ~multiple.condition.mask).any())
    assert int(multiple.condition.mask.sum()) > int(multiple.observed_mask.sum())


def test_log_uniform_count_reaches_one_and_more_than_three_planes() -> None:
    generator = torch.Generator().manual_seed(41)

    counts = {_log_uniform_count(8, generator) for _ in range(256)}

    assert 1 in counts
    assert any(count > 3 for count in counts)


def test_cross_axis_planes_remain_coherent_after_observed_overlay() -> None:
    size = 8
    prior = _prior(size=size, volume_count=1)
    labels = torch.zeros(1, size, size, size, dtype=torch.long)
    image = torch.ones(1, size, size, dtype=torch.long)
    observed = _observed(image, axis=0, index=6, size=size)
    prior.record(_prediction(labels), observed, domain=0)

    with (
        patch("src.train.prior._log_uniform_count", return_value=4),
        patch.object(
            prior,
            "_plane_candidates",
            return_value=[(0, 0), (1, 2), (2, 4)],
        ),
    ):
        sampled = prior.sample_condition(
            0,
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    stored = prior.volumes(0, 0)[0]
    expected = stored.clone()
    expected.select(0, 6).fill_(1)
    selected = sampled.condition.mask[0, 0]
    assert sampled.condition.planes == 4
    assert sampled.condition.conflicts == 0
    assert torch.equal(sampled.condition.target[0][selected], expected[selected])
    assert all(bool(sampled.condition.axis_masks[0, axis].any()) for axis in range(3))
    assert {(value.axis, value.index) for value in sampled.references[0]} == {
        (0, 0),
        (1, 2),
        (2, 4),
    }
    assert all(not bool(value.values.any()) for value in sampled.references[0])


def test_refresh_is_fifo_and_keeps_storage_bounded() -> None:
    size = 4
    prior = _prior(size=size, volume_count=2, num_phases=3)
    image = torch.zeros(1, 2, 2, dtype=torch.long)
    observed = _observed(
        image,
        axis=0,
        index=2,
        position=(1, 1),
        size=size,
        num_phases=3,
    )

    for value in (0, 1):
        labels = torch.full((1, size, size, size), value, dtype=torch.long)
        prior.record(_prediction(labels, num_phases=3), observed, domain=0)
    initial_bytes = prior.storage_bytes

    labels = torch.full((1, size, size, size), 2, dtype=torch.long)
    prior.refresh(_prediction(labels, num_phases=3), observed, domain=0)

    volumes = prior.volumes(0, 0)
    assert len(volumes) == prior.count == 2
    assert [int(volume[0, 0, 0]) for volume in volumes] == [1, 2]
    assert prior.storage_bytes == initial_bytes == 2 * (size**3 + image.numel())
    assert prior.updates == 1


def test_owned_axis_does_not_borrow_while_its_domain_bank_is_incomplete() -> None:
    size = 4
    prior = ConditionalPrior(
        num_phases=2,
        num_domains=2,
        patch_size=size,
        volume_count=2,
        plane_stride=2,
        owned_axes={0: (0,), 1: (0,)},
    )
    image = torch.zeros(1, size, size, dtype=torch.long)
    observed = _observed(image, axis=0, index=2, size=size)
    zeros = torch.zeros(1, size, size, size, dtype=torch.long)
    ones = torch.ones_like(zeros)
    prior.record(_prediction(zeros), observed, domain=0)
    prior.record(_prediction(ones), observed, domain=1)
    prior.record(_prediction(ones), observed, domain=1)

    assert not prior._banks[0].ready
    assert prior._banks[1].ready
    assert prior.volumes(0, 0) == ()


def test_plane_candidates_use_only_axes_present_in_the_training_data() -> None:
    prior = ConditionalPrior(
        num_phases=2,
        num_domains=1,
        patch_size=8,
        volume_count=1,
        plane_stride=2,
        owned_axes={0: (0,)},
    )
    observed = PlaneAnchor(torch.zeros(8, 8, dtype=torch.uint8), axis=0, index=4)

    candidates = prior._plane_candidates(
        observed,
        torch.Generator().manual_seed(1),
    )

    assert candidates
    assert {axis for axis, _ in candidates} == {0}


def _prior(
    *,
    size: int,
    volume_count: int,
    num_phases: int = 2,
) -> ConditionalPrior:
    return ConditionalPrior(
        num_phases=num_phases,
        num_domains=1,
        patch_size=size,
        volume_count=volume_count,
        plane_stride=2,
        owned_axes={0: (0, 1, 2)},
    )


def _observed(
    image: torch.Tensor,
    *,
    axis: int,
    index: int,
    size: int,
    position: tuple[int, int] | None = None,
    num_phases: int = 2,
):
    condition = build_anchors(
        (PlaneAnchor(image, axis=axis, index=index, position=position),),
        batch_size=image.shape[0],
        num_phases=num_phases,
        volume_size=size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition is not None
    return condition


def _prediction(labels: torch.Tensor, *, num_phases: int = 2) -> torch.Tensor:
    return (
        F.one_hot(labels, num_classes=num_phases)
        .movedim(-1, 1)
        .to(torch.float32)
        .mul(2.0)
        .sub(1.0)
    )
