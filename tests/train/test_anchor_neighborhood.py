from dataclasses import replace

import pytest
import torch
from anchor_neighborhood_reference import (
    compute_anchor_neighborhood_loss as reference_loss,
)

from src.anchor import PlaneAnchor, encode_anchors
from src.prepare.resize import phase_channels
from src.train.loss.anchor_neighborhood import compute_anchor_neighborhood_loss


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


def seam_volume(axis=0, soft=False, start=4, stop=7):
    labels = torch.arange(8).remainder(2)[:, None].expand(8, 8)
    planes = labels[None].expand(12, 8, 8).clone()
    planes[start:stop] = 1 - labels
    volume = phase_channels(planes.unsqueeze(0), 2).movedim(2, axis + 2)
    return volume * 0.8 + 0.1 if soft else volume


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("soft", (False, True))
@pytest.mark.parametrize("start,stop,distance", ((4, 7, 1), (3, 8, 2), (2, 9, 3)))
def test_seam_is_penalized_at_any_distance_from_the_anchor(
    axis, soft, start, stop, distance
):
    # A loss window leaves its first outside pair free, and the generator moved the
    # seam there.
    volume = seam_volume(axis, soft, start, stop)
    loss, logs = compute_anchor_neighborhood_loss(
        volume, make_condition(volume, axis=axis)
    )
    # Both seam pairs flip every voxel; the other pairs change 0 (soft: 0.18).
    assert loss.item() == pytest.approx(0.64 if soft else 1.0, abs=1e-6)
    assert logs["anchor_neighborhood/peak_distance"] == distance
    assert logs["anchor_neighborhood/tolerance"] == pytest.approx(0, abs=1e-7)
    assert logs["anchor_neighborhood/pairs"] == 11 * 64


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("soft", (False, True))
def test_uniform_change_rates_have_zero_loss_and_gradient(axis, soft):
    volume = seam_volume(axis, soft)
    plane = volume.select(axis + 2, 0).unsqueeze(axis + 2)
    volume = plane.expand_as(volume).clone().requires_grad_()
    loss, logs = compute_anchor_neighborhood_loss(
        volume, make_condition(volume, axis=axis)
    )
    assert loss.item() == pytest.approx(0, abs=1e-7)
    assert logs["anchor_neighborhood/deviation"] == pytest.approx(0, abs=1e-7)
    loss.backward()
    assert volume.grad.abs().sum() == 0


def test_gradient_reaches_only_seam_pairs_in_anchor_columns():
    volume = seam_volume(soft=True).requires_grad_()
    condition = make_condition(volume, position=(2, 1), patch=(4, 4))
    condition = replace(condition, image=condition.image.detach().requires_grad_())
    loss, logs = compute_anchor_neighborhood_loss(volume, condition)
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
    loss, logs = compute_anchor_neighborhood_loss(volume, condition)
    assert logs["anchor_neighborhood/pairs"] == 10 * 64
    loss.backward()
    assert volume.grad[:, :, 3].abs().sum() > 0
    assert volume.grad[:, :, 4:6].abs().sum() == 0


def test_overlapping_regions_do_not_duplicate_pairs():
    volume = seam_volume()
    condition = make_condition(volume)
    duplicated = replace(condition, regions=condition.regions * 2)
    loss, logs = compute_anchor_neighborhood_loss(volume, condition)
    actual, actual_logs = compute_anchor_neighborhood_loss(volume, duplicated)
    torch.testing.assert_close(actual, loss)
    for key, value in logs.items():
        torch.testing.assert_close(actual_logs[key], value)


@pytest.mark.parametrize("length", (1, 2))
def test_axis_without_two_pairs_has_zero_loss(length):
    volume = torch.rand(1, 2, length, 8, 8).softmax(1).requires_grad_()
    condition = make_condition(volume, (0,))
    loss, logs = compute_anchor_neighborhood_loss(volume, condition)
    assert loss == 0
    assert logs["anchor_neighborhood/pairs"] == (length - 1) * 64
    loss.backward()
    assert volume.grad.abs().sum() == 0


def test_reference_is_separate_for_each_volume_and_cfg_dropout_is_respected():
    volume = torch.cat((seam_volume(), torch.full_like(seam_volume(), 0.5)))
    condition = make_condition(volume)
    loss, logs = compute_anchor_neighborhood_loss(volume, condition)
    single, _ = compute_anchor_neighborhood_loss(volume[:1], make_condition(volume[:1]))
    torch.testing.assert_close(loss, single / 2)
    volume.requires_grad_()
    visible = condition.visible(torch.tensor([True, False]))
    visible_loss, visible_logs = compute_anchor_neighborhood_loss(volume, visible)
    torch.testing.assert_close(visible_loss, single)
    assert visible_logs["anchor_neighborhood/pairs"] == 11 * 64
    assert logs["anchor_neighborhood/pairs"] == 2 * 11 * 64
    visible_loss.backward()
    assert volume.grad[1].abs().sum() == 0
    hidden = condition.visible(torch.tensor([False, False]))
    hidden_loss, hidden_logs = compute_anchor_neighborhood_loss(volume, hidden)
    assert hidden_loss == 0
    assert hidden_logs["anchor_neighborhood/pairs"] == 0


@pytest.mark.parametrize("last,penalty", ((8, 0), (12, 3)))
def test_change_rate_spread_supplies_a_noise_allowance(last, penalty):
    # Pair p toggles flips[p] of 64 voxels: median 5 flips, median deviation 1 flip.
    flips = (4, 5, 6, 5, 4, 6, 5, 4, 6, 5, last)
    planes = [torch.zeros(64, dtype=torch.long)]
    for count in flips:
        plane = planes[-1].clone()
        plane[:count] = 1 - plane[:count]
        planes.append(plane)
    labels = torch.stack(planes).reshape(1, 12, 8, 8)
    volume = phase_channels(labels, 2)
    loss, logs = compute_anchor_neighborhood_loss(volume, make_condition(volume))
    assert logs["anchor_neighborhood/tolerance"] == pytest.approx(4 / 64)
    # The last pair is the only one at distance 5.
    assert loss.item() == pytest.approx(penalty / 64, abs=1e-6)


def test_half_precision_large_pair_counts_stay_finite():
    volume = seam_volume(soft=True).repeat_interleave(32, 3).repeat_interleave(32, 4)
    volume = volume.half().requires_grad_()
    loss, logs = compute_anchor_neighborhood_loss(volume, make_condition(volume))
    assert loss > 0.3
    assert logs["anchor_neighborhood/pairs"] == 11 * 256**2
    assert all(torch.isfinite(value) for value in logs.values())
    loss.backward()
    assert torch.isfinite(volume.grad).all()
    assert volume.grad.abs().sum() > 0


def compare_reference(volume, condition):
    expected_input = volume.detach().clone().requires_grad_()
    actual_input = volume.detach().clone().requires_grad_()
    expected, expected_logs = reference_loss(expected_input, condition)
    actual, actual_logs = compute_anchor_neighborhood_loss(actual_input, condition)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    assert actual_logs.keys() == expected_logs.keys()
    for key in expected_logs:
        # Counts are exact; floating reductions may differ in summation order.
        atol = 0 if key.endswith(("pairs", "axes")) else 2e-7
        rtol = 0 if atol == 0 else 2e-5
        torch.testing.assert_close(
            actual_logs[key], expected_logs[key], atol=atol, rtol=rtol, msg=key
        )
        assert not actual_logs[key].requires_grad
    expected.backward()
    actual.backward()
    # A nonzero FP16/BF16 gradient can round to the adjacent representable value.
    rtol = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[
        volume.dtype
    ]
    torch.testing.assert_close(
        actual_input.grad, expected_input.grad, atol=2e-8, rtol=rtol
    )
    assert condition.image.grad is None
    return actual, actual_input.grad


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_batched_loss_matches_loop_values_gradients_and_metrics(axis, dtype):
    rng = torch.Generator().manual_seed(527)
    plane = torch.randn(2, 3, 1, 7, 9, generator=rng).softmax(1)
    volume = plane.expand(-1, -1, 15, -1, -1).clone()
    volume[:, :, 4:8] = volume[:, :, 4:8].roll(1, 1)
    volume = volume.movedim(2, axis + 2)
    condition = make_condition(
        volume, indices=(5, 6), axis=axis, position=(1, 2), patch=(5, 6)
    )
    # Include a crossing, fixed plane and unequal per-volume visible roots.
    cross_axis = (axis + 1) % 3
    cross = make_condition(volume, indices=(2,), axis=cross_axis)
    mask = condition.mask | cross.mask
    image = torch.where(condition.mask, condition.image, cross.image)
    roots = condition.axis_masks | cross.axis_masks
    roots[1, axis].flatten()[::3] = False
    condition = replace(
        condition,
        image=image.to(dtype).detach().requires_grad_(),
        mask=mask,
        axis_masks=roots,
        regions=condition.regions * 2 + cross.regions,
    )
    loss, gradient = compare_reference(volume.to(dtype), condition)
    assert loss > 0
    assert gradient.abs().sum() > 0
    assert gradient.masked_select(mask.expand_as(gradient)).abs().sum() == 0


@pytest.mark.parametrize("length", (1, 2, 4, 9))
@pytest.mark.parametrize("active", ((0, 1), (0,), ()))
def test_batched_loss_matches_loop_for_boundaries_and_dropout(length, active):
    rng = torch.Generator().manual_seed(83)
    volume = torch.randn(2, 3, length, 4, 5, generator=rng).softmax(1)
    condition = make_condition(volume, indices=(0, length - 1))
    condition = replace(condition, active_batches=active)
    compare_reference(volume, condition)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_batched_loss_matches_loop_under_cuda_autocast():
    volume = seam_volume(soft=True).cuda().half()
    condition = make_condition(volume)
    with torch.autocast("cuda", dtype=torch.float16):
        compare_reference(volume, condition)
