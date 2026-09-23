from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from src.anchor import AnchorRegion, PlaneAnchor, encode_anchors
from src.data.slice import AnchorTripletSampler, sample_pairs


def test_focused_pairs_use_active_anchor_batches():
    volume = torch.zeros(2, 2, 8, 8, 8)
    volume[1, 0, :, 5, 6] = 1
    condition = encode_anchors(
        [PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 1, 5)],
        2,
        2,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    condition = replace(condition, active_batches=(1,))

    def choose(high, size, *args, **kwargs):
        if high == 16 and size == (4,):
            return torch.tensor([0, 8, 8, 8])
        if high == 8 and size == ():
            return torch.tensor(6)
        return torch.zeros(size, dtype=torch.long)

    with patch("src.data.slice.torch.randint", side_effect=choose):
        selected, _ = sample_pairs(volume, volume, 0, 4, 3, anchor=condition)

    assert torch.equal(selected[:2, 0].amax((-2, -1)), torch.ones(2))
    assert torch.equal(selected[2:, 0], torch.zeros_like(selected[2:, 0]))


def test_focused_pairs_skip_measured_plane_and_empty_anchor_region():
    volume = torch.zeros(2, 2, 8, 8, 8)
    volume[1, 0, 3, 5, 6] = 1
    volume[1, 0, 2] = 2
    volume.requires_grad_()
    anchor = encode_anchors(
        [PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 1, 5)],
        2, 2, 8, torch.device("cpu"), torch.float32,
    )
    anchor = replace(
        anchor,
        regions=(
            AnchorRegion(1, 1, 2, 4, 1, 1),
            AnchorRegion(1, 5, 3, 6, 1, 1),
        ),
        active_batches=(1,),
    )
    measured = encode_anchors(
        [PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, 2)],
        2, 2, 8, torch.device("cpu"), torch.float32,
    )
    measured = replace(measured, active_batches=(1,))

    def choose(high, size, *args, **kwargs):
        if high == 15 and size == (4,):
            return torch.tensor([0, 8, 8, 8])
        return torch.zeros(size, dtype=torch.long)

    with patch("src.data.slice.torch.randint", side_effect=choose):
        selected, _ = sample_pairs(
            volume, volume, 0, 4, 3, anchor=anchor, measured=measured,
        )

    assert torch.equal(selected[:2, 0].amax((-2, -1)), torch.ones(2))
    assert selected.max() == 1
    selected.sum().backward()
    assert volume.grad[1, :, 2].count_nonzero() == 0


def test_connectivity_samples_every_parallel_anchor_and_includes_adjacent_gap():
    anchors = tuple(
        PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, i) for i in (2, 5)
    )
    condition = encode_anchors(anchors, 1, 2, 8, torch.device("cpu"), torch.float32)
    volume = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    sampler = AnchorTripletSampler(max_gap=3, windows_per_plane=4)
    located = sampler._sample_anchor_triplets(volume, condition)
    for index in (2, 5):
        slots = [i for i, pos in enumerate(located.locations) if pos == (0, 0, index)]
        assert len(slots) == 4
        assert located.triplets.gaps[slots[0]] == 1
    assert located.triplets.values.shape[-2:] == (4, 4)
    real, fake = sampler.sample(volume, volume.detach(), condition)
    torch.testing.assert_close(real.values, fake.values)
    fake.values.sum().backward()
    assert volume.grad.abs().sum() > 0


@pytest.mark.parametrize("axis", range(3))
def test_measured_parallel_plane_is_excluded_from_adversarial_gradients(axis):
    volume = torch.randn(1, 2, 5, 5, 5, requires_grad=True)
    condition = encode_anchors(
        [PlaneAnchor(torch.zeros(5, 5, dtype=torch.long), axis, 2)],
        1,
        2,
        5,
        torch.device("cpu"),
        torch.float32,
    )
    previous, _ = sample_pairs(volume, volume, axis, 100, 5, measured=condition)
    previous.sum().backward()
    assert volume.grad.select(axis + 2, 2).count_nonzero() == 0
    assert volume.grad.sum() > 0


def test_triplet_sampler_uses_integer_regions_without_dense_mask_scan():
    prediction = torch.zeros(1, 2, 8, 8, 8)
    condition = encode_anchors(
        [PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, 3)],
        1,
        2,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    with (
        patch.object(
            torch.Tensor, "nonzero", side_effect=AssertionError("dense mask scan")
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
    ):
        real, fake = AnchorTripletSampler(windows_per_plane=4).sample(
            prediction, prediction, condition
        )
    assert len(real) == len(fake) == 12
