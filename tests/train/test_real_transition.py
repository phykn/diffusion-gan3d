from types import SimpleNamespace

import pytest
import torch

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
        connectivity_max_gap=1,
        slice_pairs_per_axis=2,
    )
    trainer.streams = {0: {0: None}, 1: {1: None}}
    trainer.diagnostics = {}
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
