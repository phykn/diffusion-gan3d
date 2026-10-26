from types import SimpleNamespace

import pytest
import torch

from src.data.augment import CriticAugment
from src.prepare.height import height_field
from src.prepare.resize import phase_channels
from src.train.batch import RealBatch
from src.train.loss.transition import (
    compute_real_transition_loss,
    phase_pairs,
    transition_statistics,
)
from src.train.trainer import Trainer


@pytest.mark.parametrize(
    "axis,directions", [(0, {"x", "y"}), (1, {"x", "z"}), (2, {"y", "z"})]
)
def test_only_observed_physical_directions_are_constrained(axis, directions):
    labels = torch.arange(5).remainder(2)[None, :, None].expand(2, -1, 5)
    images = phase_channels(labels, 2)
    loss, diagnostics = compute_real_transition_loss({axis: images}, {axis: images}, 2)
    assert loss == 0
    assert {key.split("/")[1] for key in diagnostics} == directions
    assert {key.split("/")[2] for key in diagnostics} == {"gap1", "gap2"}
    wrong, _ = compute_real_transition_loss(
        {axis: images}, {axis: images.transpose(2, 3)}, 1
    )
    assert wrong > 0


def test_joint_probabilities_preserve_soft_phase_fractions():
    image = torch.tensor([0.25, 0.75])[None, :, None, None].expand(1, -1, 4, 6)
    expected = torch.tensor([[0.0625, 0.1875], [0.1875, 0.5625]])
    for dim in (2, 3):
        for gap in (1, 2):
            stats = phase_pairs(image, dim, gap)
            torch.testing.assert_close(stats, expected.expand_as(stats))
            torch.testing.assert_close(stats.sum((-2, -1)), torch.ones(stats.shape[:2]))


def test_statistics_do_not_require_pixelwise_correspondence():
    labels = torch.stack(
        (torch.zeros(5, 5, dtype=torch.long), torch.ones(5, 5, dtype=torch.long))
    )
    real = phase_channels(labels, 2)
    fake = real.flip(0)
    assert (real - fake).abs().mean() == 1
    loss, _ = compute_real_transition_loss({0: real}, {0: fake}, 3)
    assert loss == 0


def test_real_statistics_are_detached_and_generated_statistics_have_gradients():
    torch.manual_seed(1)
    real = phase_channels(torch.randint(2, (2, 5, 5)), 2).requires_grad_()
    logits = torch.randn(2, 2, 5, 5, requires_grad=True)
    loss, _ = compute_real_transition_loss({1: real}, {1: logits.softmax(1)}, 2)
    loss.backward()
    assert real.grad is None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_height_disjoint_crops_do_not_supply_targets():
    real = torch.zeros(1, 2, 4, 4)
    real[:, 0] = 1
    fake = torch.full_like(real, 0.5, requires_grad=True)
    low = height_field((4, 4), 0, 0, 1, 16)
    high = height_field((4, 4), 0, 8, 1, 16)
    loss, diagnostics = compute_real_transition_loss(
        {1: real}, {1: fake}, 2, {1: low}, {1: high}
    )
    assert loss == 0
    assert all(
        value == 0 for key, value in diagnostics.items() if key.endswith("matches")
    )
    loss.backward()
    assert fake.grad.abs().sum() == 0


def test_height_matching_rejects_wrong_height_but_accepts_matching_subcrop():
    labels = torch.zeros(1, 8, 4, dtype=torch.long)
    labels[:, 4:] = 1
    real = phase_channels(labels, 2)
    fake = real[:, :, 4:]
    full = height_field((8, 4), 0, 0, 1, 16)
    high = height_field((4, 4), 0, 4, 1, 16)
    low = height_field((4, 4), 0, 0, 1, 16)
    matched, _ = compute_real_transition_loss(
        {1: real}, {1: fake}, 2, {1: full}, {1: high}
    )
    misplaced, _ = compute_real_transition_loss(
        {1: real}, {1: fake}, 2, {1: full}, {1: low}
    )
    assert matched == 0
    assert misplaced > 0


def test_xy_without_observed_height_is_not_used_for_height_targets():
    images = torch.full((1, 2, 4, 4), 0.5, requires_grad=True)
    loss, diagnostics = compute_real_transition_loss(
        {0: images}, {0: images}, 2, {}, {}
    )
    assert loss == 0 and diagnostics == {}
    loss.backward()
    assert images.grad.abs().sum() == 0


def test_gaps_larger_than_available_image_are_skipped():
    images = torch.full((1, 2, 2, 3), 0.5)
    loss, diagnostics = compute_real_transition_loss({0: images}, {0: images}, 8)
    assert loss == 0
    assert set(diagnostics) == {
        f"real_transition/{direction}/gap{gap}{suffix}"
        for direction, gap in (("y", 1), ("x", 1), ("x", 2))
        for suffix in ("", "/matches")
    }


def test_absent_height_targets_remain_finite_with_half_precision():
    images = torch.full((1, 2, 256, 256), 0.5, dtype=torch.float16, requires_grad=True)
    loss, _ = compute_real_transition_loss({0: images}, {0: images}, 1, {}, {})
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert images.grad.abs().sum() == 0


def test_transition_targets_exclude_borrowed_domain_planes():
    trainer = object.__new__(Trainer)
    trainer.settings = SimpleNamespace(
        loss=SimpleNamespace(real_transition_weight=1.0),
        real_transition_max_gap=1,
        slice_pairs_per_plane=2,
    )
    trainer.streams = {0: {0: None}, 1: {1: None}}
    trainer.diagnostics = {}
    trainer.critic_augment = CriticAugment()
    volume = torch.zeros(1, 2, 4, 4, 4)
    volume[:, 0] = 1
    own = volume[:, :, 0]
    borrowed = own.flip(1)
    prepared = SimpleNamespace(
        transition=0,
        domain=0,
        model_conditions={},
        real=RealBatch(images={0: own, 1: borrowed}, domains={0: 0, 1: 1}),
    )
    loss = trainer._real_transition_loss(prepared, volume)
    assert loss == 0
    assert not any("/z/" in key for key in trainer.diagnostics)


@pytest.mark.parametrize("height_enabled", (False, True))
def test_cached_statistics_preserve_real_loss_and_generated_gradients(height_enabled):
    torch.manual_seed(7)
    real = {axis: torch.rand(2, 2, 8, 8).softmax(1) for axis in range(3)}
    fake = {axis: torch.rand(2, 2, 8, 8, requires_grad=True) for axis in range(3)}
    heights = (
        {axis: height_field((8, 8), 0, [0, 1], 1, 16) for axis in (1, 2)}
        if height_enabled
        else None
    )
    stats = transition_statistics(real, 3, heights)
    original, original_logs = compute_real_transition_loss(
        real, fake, 3, heights, heights
    )
    cached, cached_logs = compute_real_transition_loss(
        real, fake, 3, heights, heights, stats
    )
    torch.testing.assert_close(original, cached, rtol=0, atol=0)
    assert original_logs.keys() == cached_logs.keys()
    for key in original_logs:
        torch.testing.assert_close(original_logs[key], cached_logs[key], rtol=0, atol=0)
    active = [fake[axis] for axis in ((1, 2) if height_enabled else (0, 1, 2))]
    first = torch.autograd.grad(original, active)
    second = torch.autograd.grad(cached, active)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rotated_planes_share_pooled_row_and_column_targets():
    labels = torch.arange(6).remainder(2)[None, :, None].expand(2, -1, 6)
    images = phase_channels(labels, 2)
    plain = transition_statistics({0: images, 1: images}, 2)
    pooled = transition_statistics({0: images, 1: images}, 2, rotations=(0,))
    for gap in (1, 2):
        rows, cols = plain[0, 2, gap].values, plain[0, 3, gap].values
        assert torch.equal(rows.mean((0, 1)), cols.mean((0, 1))) == (gap == 2)
        transposed = phase_pairs(images.transpose(2, 3), 2, gap)
        for dim in (2, 3):
            torch.testing.assert_close(
                pooled[0, dim, gap].values, torch.cat((rows, transposed), 1)
            )
            torch.testing.assert_close(
                pooled[1, dim, gap].values, plain[1, dim, gap].values
            )
        torch.testing.assert_close(
            pooled[0, 2, gap].values.mean((0, 1)),
            (rows.mean((0, 1)) + cols.mean((0, 1))) * 0.5,
        )


def test_rotated_targets_apply_to_the_real_transition_loss():
    labels = torch.arange(6).remainder(2)[None, :, None].expand(2, -1, 6)
    images = phase_channels(labels, 2)
    exact, _ = compute_real_transition_loss({0: images}, {0: images}, 1)
    pooled, diagnostics = compute_real_transition_loss(
        {0: images}, {0: images}, 1, rotations=(0,)
    )
    rotated, _ = compute_real_transition_loss(
        {0: images}, {0: images.transpose(2, 3)}, 1, rotations=(0,)
    )
    assert exact == 0
    assert pooled > 0
    torch.testing.assert_close(rotated, pooled)
    torch.testing.assert_close(
        diagnostics["real_transition/y/gap1"], diagnostics["real_transition/x/gap1"]
    )


def test_height_conditioned_statistics_ignore_rotation():
    real = {1: torch.rand(2, 2, 6, 6).softmax(1)}
    heights = {1: height_field((6, 6), 0, [0, 1], 1, 16)}
    plain = transition_statistics(real, 2, heights)
    rotated = transition_statistics(real, 2, heights, rotations=(1,))
    assert plain.keys() == rotated.keys()
    for key, stats in plain.items():
        torch.testing.assert_close(rotated[key].values, stats.values)
        torch.testing.assert_close(rotated[key].low, stats.low)


@pytest.mark.parametrize(
    "planes,probability,expected",
    [
        (None, 1.0, ()),
        ({"xy": {"flip_axes": ["x"], "rotate_90": True}}, 0.0, ()),
        (
            {
                "xy": {"flip_axes": ["x", "y"], "rotate_90": True},
                "xz": {"flip_axes": ["x", "z"], "rotate_90": False},
                "yz": {"flip_axes": [], "rotate_90": True},
            },
            0.5,
            (0, 2),
        ),
    ],
)
def test_critic_augmentation_reports_rotated_planes(planes, probability, expected):
    assert CriticAugment(planes, probability).rotated_axes == expected


def test_trainer_transition_targets_use_rotating_augmentation_planes():
    trainer = object.__new__(Trainer)
    trainer.settings = SimpleNamespace(
        loss=SimpleNamespace(real_transition_weight=1.0),
        real_transition_max_gap=1,
        slice_pairs_per_plane=4,
    )
    trainer.streams = {0: {0: None}}
    trainer.diagnostics = {}
    trainer.critic_augment = CriticAugment({"xy": {"flip_axes": [], "rotate_90": True}})
    labels = torch.arange(4).remainder(2)[None, :, None].expand(2, -1, 4)
    images = phase_channels(labels, 2)
    prepared = SimpleNamespace(
        transition=0,
        domain=0,
        model_conditions={},
        real=RealBatch(images={0: images}, domains={0: 0}),
    )
    statistics = trainer._transition_statistics(prepared)
    torch.testing.assert_close(statistics[0, 2, 1].values, statistics[0, 3, 1].values)
    volume = images[:1, :, None].expand(-1, -1, 4, -1, -1)
    assert trainer._real_transition_loss(prepared, volume, statistics) > 0
    trainer.critic_augment = CriticAugment()
    assert trainer._real_transition_loss(prepared, volume) == 0
