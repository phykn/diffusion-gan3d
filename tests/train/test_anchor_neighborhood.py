from dataclasses import replace

import pytest
import torch

from src.anchor import PlaneAnchor, encode_anchors
from src.prepare.resize import phase_channels
from src.train.loss.anchor_neighborhood import compute_anchor_neighborhood_loss
from src.train.trainer import Trainer


def make_condition(volume, indices=(5,), axis=0, position=None, patch=None):
    images = [volume.select(axis + 2, index) for index in indices]
    if patch is not None:
        images = [image[..., : patch[0], : patch[1]] for image in images]
    return encode_anchors(
        [
            PlaneAnchor(image, axis, index, position)
            for image, index in zip(images, indices)
        ],
        len(volume),
        volume.shape[1],
        volume.shape[2:],
        volume.device,
        volume.dtype,
    )


def seam_volume(axis=0, soft=False):
    labels = torch.arange(8).remainder(2)[:, None].expand(8, 8)
    planes = labels[None].expand(12, 8, 8).clone()
    planes[4:7] = 1 - labels
    volume = phase_channels(planes.unsqueeze(0), 2).movedim(2, axis + 2)
    return volume * 0.8 + 0.1 if soft else volume


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("soft", (False, True))
def test_copied_neighbor_exposes_the_seam_one_pair_beyond_the_anchor(axis, soft):
    volume = seam_volume(axis, soft)
    condition = make_condition(volume, axis=axis)
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    assert loss.item() == pytest.approx(0.32 if soft else 0.5, abs=1e-6)
    assert logs["anchor_neighborhood/distance0/loss"] == pytest.approx(0, abs=1e-7)
    assert logs["anchor_neighborhood/distance1/loss"] > 0.6
    assert logs["anchor_neighborhood/distance1/tv"] == pytest.approx(
        0.64 if soft else 1.0
    )
    assert logs["anchor_neighborhood/distance1/tolerance"] == pytest.approx(0, abs=1e-7)
    assert logs["anchor_neighborhood/distance0/pairs"] == 128
    assert logs["anchor_neighborhood/distance1/pairs"] == 128
    assert logs["anchor_neighborhood/far_pairs"] == 7 * 64


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("soft", (False, True))
def test_uniform_joint_statistics_have_zero_loss_and_gradient(axis, soft):
    volume = seam_volume(axis, soft)
    plane = volume.select(axis + 2, 0).unsqueeze(axis + 2)
    volume = plane.expand_as(volume).clone().requires_grad_()
    loss, logs = compute_anchor_neighborhood_loss(
        volume, make_condition(volume, axis=axis), 2
    )
    assert loss.item() == pytest.approx(0, abs=1e-7)
    assert logs["anchor_neighborhood/distance1/tv"] == pytest.approx(0, abs=1e-7)
    loss.backward()
    assert volume.grad.abs().sum() == 0


def test_gradient_only_reaches_unfixed_nearby_voxels_and_reference_is_detached():
    volume = seam_volume(soft=True).requires_grad_()
    condition = make_condition(volume, position=(2, 1), patch=(4, 4))
    condition = replace(condition, image=condition.image.detach().requires_grad_())
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    loss.backward()
    allowed = torch.zeros_like(volume, dtype=torch.bool)
    allowed[:, :, (3, 4, 6, 7), 2:6, 1:5] = True
    assert volume.grad[allowed].abs().sum() > 0
    assert volume.grad[~allowed].abs().sum() == 0
    assert condition.image.grad is None
    assert all(not value.requires_grad for value in logs.values())


def test_fixed_endpoint_is_included_but_two_fixed_endpoints_are_excluded():
    volume = seam_volume(soft=True).requires_grad_()
    condition = make_condition(volume, (4, 5))
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    assert logs["anchor_neighborhood/distance0/pairs"] == 128
    assert logs["anchor_neighborhood/distance1/pairs"] == 128
    assert logs["anchor_neighborhood/distance0/tv"] > 0.3
    loss.backward()
    assert volume.grad[:, :, 3].abs().sum() > 0
    assert volume.grad[:, :, 4:6].abs().sum() == 0


def test_overlapping_regions_do_not_duplicate_pairs():
    volume = seam_volume()
    condition = make_condition(volume)
    duplicated = replace(condition, regions=condition.regions * 2)
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    actual, actual_logs = compute_anchor_neighborhood_loss(volume, duplicated, 2)
    torch.testing.assert_close(actual, loss)
    for key, value in logs.items():
        torch.testing.assert_close(actual_logs[key], value)


@pytest.mark.parametrize("length", (1, 3, 5))
def test_axis_without_far_pairs_is_skipped(length):
    volume = torch.full((1, 2, length, 8, 8), 0.5, requires_grad=True)
    condition = make_condition(volume, (length // 2,))
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    assert loss == 0
    assert logs["anchor_neighborhood/reference_axes"] == 0
    assert logs["anchor_neighborhood/distance0/pairs"] == 0
    loss.backward()
    assert volume.grad.abs().sum() == 0


def test_far_reference_is_separate_for_each_volume_and_cfg_dropout_is_respected():
    volume = torch.cat((seam_volume(), torch.full_like(seam_volume(), 0.5)))
    condition = make_condition(volume)
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    single, _ = compute_anchor_neighborhood_loss(
        volume[:1], make_condition(volume[:1]), 2
    )
    assert loss == single / 2
    volume.requires_grad_()
    visible = Trainer.visible_anchor(condition, torch.tensor([True, False]))
    visible_loss, visible_logs = compute_anchor_neighborhood_loss(volume, visible, 2)
    assert visible_loss == single
    assert visible_logs["anchor_neighborhood/distance0/pairs"] == 128
    assert logs["anchor_neighborhood/distance0/pairs"] == 256
    visible_loss.backward()
    assert volume.grad[1].abs().sum() == 0
    hidden = Trainer.visible_anchor(condition, torch.tensor([False, False]))
    hidden_loss, hidden_logs = compute_anchor_neighborhood_loss(volume, hidden, 2)
    assert hidden_loss == 0
    assert hidden_logs["anchor_neighborhood/far_pairs"] == 0


def test_far_pair_variation_supplies_a_leave_one_out_noise_allowance():
    labels = torch.zeros(1, 12, 8, 8, dtype=torch.long)
    labels[:, 1::2] = 1
    volume = phase_channels(labels, 2).requires_grad_()
    condition = make_condition(volume)
    loss, logs = compute_anchor_neighborhood_loss(volume, condition, 2)
    assert logs["anchor_neighborhood/distance0/tv"] == pytest.approx(0.5)
    assert logs["anchor_neighborhood/distance0/tolerance"] == pytest.approx(4 / 7)
    assert loss == 0
    loss.backward()
    assert volume.grad.abs().sum() == 0


def test_single_far_pair_can_supply_a_reference():
    volume = torch.zeros(1, 2, 4, 8, 8)
    volume[:, 0] = 1
    volume[:, :, :2] = volume[:, :, :2].flip(1)
    loss, logs = compute_anchor_neighborhood_loss(
        volume, make_condition(volume, (0,)), 2
    )
    assert loss > 0
    assert logs["anchor_neighborhood/far_pairs"] == 64
    assert logs["anchor_neighborhood/reference_axes"] == 1


def test_half_precision_large_pair_counts_stay_finite():
    volume = seam_volume(soft=True).repeat_interleave(32, 3).repeat_interleave(32, 4)
    volume = volume.half().requires_grad_()
    loss, logs = compute_anchor_neighborhood_loss(volume, make_condition(volume), 2)
    assert loss > 0.3
    assert logs["anchor_neighborhood/distance1/pairs"] == 2 * 256**2
    assert all(torch.isfinite(value) for value in logs.values())
    loss.backward()
    assert torch.isfinite(volume.grad).all()
    assert volume.grad.abs().sum() > 0
