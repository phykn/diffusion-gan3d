from dataclasses import replace

import pytest
import torch

from src.anchor import PlaneAnchor, encode_anchors
from src.prepare.height import height_field
from src.prepare.resize import phase_channels
from src.train.loss.anchor_continuity import compute_anchor_continuity_loss
from src.train.loss.transition import transition_statistics


def condition(image, axis=0, index=3, size=8, position=None):
    return encode_anchors(
        (PlaneAnchor(image, axis, index, position),),
        1,
        image.shape[1] if image.ndim == 4 else 2,
        size,
        torch.device("cpu"),
        torch.float32,
    )


def reference(axis=0, gap=1):
    # 16 adjacent pairs: per phase, six stay and two switch; T = [.75, .25].
    labels = torch.arange(17).div(4, rounding_mode="floor").remainder(2)
    images = phase_channels(labels[None, :, None].expand(2, -1, 8), 2)
    plane = 1 if axis == 0 else 0
    if axis == 2:
        images = images.transpose(-1, -2)
    return transition_statistics({plane: images}, gap)


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("soft", (False, True))
def test_matched_transitions_beat_detached_and_extruded_neighbors(axis, soft):
    labels = torch.arange(8).remainder(2)[None].expand(8, -1)
    anchor = condition(labels, axis)
    root = phase_channels(labels.unsqueeze(0), 2)
    matched = root * 0.75 + (1 - root) * 0.25
    detached = torch.full_like(root, 0.5)
    if not soft:
        matched, detached = root.clone(), root.clone()
        matched[:, :, :2] = 1 - root[:, :, :2]
        detached[:, :, :4] = 1 - root[:, :, :4]
    results = {}
    for name, neighbor in (
        ("matched", matched),
        ("detached", detached),
        ("extruded", root),
    ):
        volume = torch.zeros(1, 2, 8, 8, 8)
        volume[:, 0] = 1
        volume.select(axis + 2, 3).copy_(root)
        for index in (2, 4):
            volume.select(axis + 2, index).copy_(neighbor)
        loss, logs = compute_anchor_continuity_loss(
            volume, anchor, anchor, reference(axis), 1
        )
        results[name] = loss.item()
        assert logs["anchor_continuity/gap1/pairs"] == 128
        assert logs["anchor_continuity/gap1/tolerance"] == 0
        assert logs["anchor_continuity/sharpness_far"] == 1
        if name == "matched":
            assert logs["anchor_continuity/sharpness_near"] == (0.625 if soft else 1)
            assert logs["anchor_continuity/sharpness_difference"] == (
                -0.375 if soft else 0
            )
    assert results["matched"] == pytest.approx(0, abs=1e-7)
    assert results["detached"] == pytest.approx(0.25)
    assert results["extruded"] == pytest.approx(0.25)


def test_target_uses_anchor_fraction_not_real_crop_fraction_and_retains_soft_phases():
    root = torch.tensor([0.8, 0.2])[None, :, None, None].expand(1, 2, 8, 8)
    anchor = condition(root)
    # Identical soft roots: J_ab = pi_a * P_b, with P = pi @ T.
    volume = torch.tensor([0.65, 0.35])[None, :, None, None, None].expand(1, 2, 8, 8, 8)
    loss, _ = compute_anchor_continuity_loss(volume, anchor, anchor, reference(), 1)
    # Soft phase independence cannot recover the conditional coupling. Its
    # nonzero TV (.16) also distinguishes this computation from argmax labels.
    assert loss == pytest.approx(0.16)
    pure = condition(torch.zeros(8, 8, dtype=torch.long))
    volume = torch.tensor([0.75, 0.25])[None, :, None, None, None].expand(1, 2, 8, 8, 8)
    loss, _ = compute_anchor_continuity_loss(volume, pure, pure, reference(), 1)
    assert loss == 0


def test_gradient_reaches_only_eligible_neighbors_and_not_observations():
    anchor = condition(torch.zeros(4, 4, dtype=torch.long), position=(2, 1))
    anchor = replace(anchor, image=anchor.image.requires_grad_())
    real = phase_channels(torch.zeros(2, 8, 8, dtype=torch.long), 2).requires_grad_()
    stats = transition_statistics({1: real}, 2)
    logits = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
    loss, _ = compute_anchor_continuity_loss(
        logits.softmax(1), anchor, anchor, stats, 2
    )
    loss.backward()
    expected = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    expected[:, :, (1, 2, 4, 5), 2:6, 1:5] = True
    assert logits.grad[expected.expand_as(logits)].abs().sum() > 0
    assert logits.grad[~expected.expand_as(logits)].abs().sum() == 0
    assert real.grad is None and anchor.image.grad is None


def test_replay_roots_are_measured_and_all_fixed_destination_voxels_are_excluded():
    image = torch.zeros(8, 8, dtype=torch.long)
    measured = condition(image)
    replay = encode_anchors(
        (PlaneAnchor(image, 0, 3), PlaneAnchor(image, 0, 4)),
        1,
        2,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    volume = torch.full((1, 2, 8, 8, 8), 0.5, requires_grad=True)
    loss, logs = compute_anchor_continuity_loss(
        volume, measured, replay, reference(), 1
    )
    assert loss > 0
    assert logs["anchor_continuity/gap1/pairs"] == 64
    loss.backward()
    assert volume.grad[:, :, 2].abs().sum() > 0
    assert volume.grad[:, :, 4:].abs().sum() == 0


def test_hidden_and_out_of_range_anchors_and_missing_directions_are_skipped():
    anchor = condition(torch.zeros(8, 8, dtype=torch.long), index=0)
    volume = torch.full((1, 2, 8, 8, 8), 0.5, requires_grad=True)
    loss, logs = compute_anchor_continuity_loss(volume, anchor, anchor, reference(), 1)
    assert loss > 0 and logs["anchor_continuity/gap1/pairs"] == 64
    hidden = replace(anchor, active_batches=(), mask=torch.zeros_like(anchor.mask))
    hidden_loss, logs = compute_anchor_continuity_loss(
        volume, hidden, hidden, reference(), 1
    )
    assert hidden_loss == 0 and logs["anchor_continuity/gap1/pairs"] == 0
    assert torch.isnan(logs["anchor_continuity/sharpness_near"])
    hidden_loss.backward()
    assert volume.grad.abs().sum() == 0
    only_xy = transition_statistics(
        {0: phase_channels(torch.zeros(1, 8, 8, dtype=torch.long), 2)}, 1
    )
    missing_loss, logs = compute_anchor_continuity_loss(
        volume, anchor, anchor, only_xy, 1
    )
    assert missing_loss == 0 and logs["anchor_continuity/gap1/pairs"] == 0


def test_crop_variation_supplies_a_data_derived_tv_allowance():
    # Half the crops always switch, half always stay, with the same phase mass.
    stripes = torch.arange(8).remainder(2)[None, :, None].expand(1, -1, 8)
    columns = stripes.transpose(1, 2)
    real = phase_channels(torch.cat((stripes, columns)), 2)
    stats = transition_statistics({1: real}, 1)
    anchor = condition(torch.zeros(8, 8, dtype=torch.long))
    volume = torch.zeros(1, 2, 8, 8, 8)
    volume[:, 0] = 1
    loss, logs = compute_anchor_continuity_loss(volume, anchor, anchor, stats, 1)
    assert logs["anchor_continuity/gap1/tv"] == 0.5
    assert logs["anchor_continuity/gap1/tolerance"] == 0.5
    assert loss == 0


def test_missing_source_phase_does_not_create_a_zero_mass_target():
    real = phase_channels(torch.zeros(2, 8, 8, dtype=torch.long), 2)
    stats = transition_statistics({1: real}, 1)
    anchor = condition(torch.ones(8, 8, dtype=torch.long))
    volume = torch.full((1, 2, 8, 8, 8), 0.5, requires_grad=True)
    loss, logs = compute_anchor_continuity_loss(volume, anchor, anchor, stats, 1)
    assert loss == 0 and logs["anchor_continuity/gap1/pairs"] == 0
    loss.backward()
    assert volume.grad.abs().sum() == 0


def test_height_matching_keeps_forward_and_reverse_transitions_separate():
    # At z=2: phase 0 below, phase 1 at and above the anchor.
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, 2:] = 1
    real = phase_channels(labels, 2)
    real_height = height_field((8, 8), 0, 0, 1, 16)
    stats = transition_statistics({1: real}, 1, {1: real_height})
    volume_height = height_field((8, 8, 8), 0, 0, 1, 16)
    anchor = condition(torch.ones(8, 8, dtype=torch.long), index=2)
    correct = phase_channels(labels[:, :, :, None].expand(1, 8, 8, 8), 2)
    loss, logs = compute_anchor_continuity_loss(
        correct, anchor, anchor, stats, 1, volume_height
    )
    assert loss == 0 and logs["anchor_continuity/gap1/pairs"] == 128
    swapped = correct.clone()
    swapped[:, :, 1], swapped[:, :, 3] = correct[:, :, 3], correct[:, :, 1]
    wrong, _ = compute_anchor_continuity_loss(
        swapped, anchor, anchor, stats, 1, volume_height
    )
    assert wrong == 1
    high = height_field((8, 8, 8), 0, 8, 1, 16)
    absent, logs = compute_anchor_continuity_loss(
        correct, anchor, anchor, stats, 1, high
    )
    assert absent == 0 and logs["anchor_continuity/gap1/pairs"] == 0


@pytest.mark.parametrize("axis", (1, 2))
def test_vertical_anchor_targets_follow_each_height_row(axis):
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, 4:] = torch.arange(8).remainder(2)
    real = phase_channels(labels, 2)
    plane = 2 if axis == 1 else 1
    stats = transition_statistics(
        {plane: real}, 1, {plane: height_field((8, 8), 0, 0, 1, 16)}
    )
    volume = torch.zeros(1, 2, 8, 8, 8)
    volume[:, 0] = 1
    # At high z the observed horizontal direction switches phase each voxel.
    for index in (2, 4):
        volume.select(axis + 2, index)[:, :, 4:] = torch.tensor([0.0, 1.0])[
            None, :, None, None
        ]
    anchor = condition(torch.zeros(8, 8, dtype=torch.long), axis)
    loss, logs = compute_anchor_continuity_loss(
        volume,
        anchor,
        anchor,
        stats,
        1,
        height_field((8, 8, 8), 0, 0, 1, 16),
    )
    assert loss == 0 and logs["anchor_continuity/gap1/pairs"] == 128
    wrong, _ = compute_anchor_continuity_loss(
        volume.flip(2),
        anchor,
        anchor,
        stats,
        1,
        height_field((8, 8, 8), 0, 0, 1, 16),
    )
    assert wrong == 1


def test_all_gaps_are_measured_and_empty_far_region_is_not_reported_as_sharp():
    anchor = condition(torch.zeros(8, 8, dtype=torch.long))
    volume = torch.full((1, 2, 8, 8, 8), 0.5)
    _, logs = compute_anchor_continuity_loss(
        volume, anchor, anchor, reference(gap=8), 8
    )
    assert {key for key in logs if key.endswith("/loss")} == {
        f"anchor_continuity/gap{gap}/loss" for gap in range(1, 8)
    }
    assert logs["anchor_continuity/near_voxels"] == 7 * 64
    assert logs["anchor_continuity/far_voxels"] == 0
    assert torch.isnan(logs["anchor_continuity/sharpness_far"])


def test_partial_fixed_neighbors_and_per_item_dropout_exclude_only_their_voxels():
    measured = encode_anchors(
        (PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, 3),),
        2,
        2,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    fixed = encode_anchors(
        (
            PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, 3),
            PlaneAnchor(torch.zeros(4, 4, dtype=torch.long), 0, 4, (0, 0)),
        ),
        2,
        2,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    measured = replace(measured, active_batches=(0,))
    fixed = replace(fixed, active_batches=(0,))
    volume = torch.full((2, 2, 8, 8, 8), 0.5, requires_grad=True)
    loss, logs = compute_anchor_continuity_loss(volume, measured, fixed, reference(), 1)
    assert logs["anchor_continuity/gap1/pairs"] == 112
    loss.backward()
    assert volume.grad[1].abs().sum() == 0
    assert volume.grad[0, :, 4, :4, :4].abs().sum() == 0
    assert volume.grad[0, :, 4, 4:, 4:].abs().sum() > 0


def test_half_precision_counts_and_loss_remain_finite():
    anchor = condition(
        torch.zeros(256, 256, dtype=torch.long), size=(3, 256, 256), index=1
    )
    volume = torch.full(
        (1, 2, 3, 256, 256), 0.5, dtype=torch.float16, requires_grad=True
    )
    loss, logs = compute_anchor_continuity_loss(volume, anchor, anchor, reference(), 1)
    assert torch.isfinite(loss)
    assert loss == 0.25
    assert logs["anchor_continuity/gap1/pairs"] == 2 * 256**2
    loss.backward()
    assert torch.isfinite(volume.grad).all() and volume.grad.abs().sum() > 0
