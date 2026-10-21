from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.model.critic import CriticScores
from src.train.loss.denoiser import adversarial_loss, profile_loss


class PairCritic(nn.Module):
    def forward(self, previous, current, time, domain):
        score = previous.flatten(1).mean(1) + 0.25 * current.flatten(1).mean(1)
        return CriticScores(score)


@pytest.mark.parametrize("diagnose", [False, True])
def test_grouped_objective_preserves_weighting_and_generator_gradient(diagnose):
    weight = torch.tensor(0.7, requires_grad=True)
    factors = {0: 1.0, 1: 2.0, 2: 4.0}
    previous = {
        axis: weight * factor * torch.ones(2, 2, 3, 3)
        for axis, factor in factors.items()
    }
    current = torch.ones(2, 2, 3, 3)
    batch = SimpleNamespace(
        fake={axis: (value, current) for axis, value in previous.items()},
        transition=1,
        critic_domains={axis: 0 for axis in factors},
        fake_heights={},
        fake_profiles={},
        logits=weight,
    )
    loss, diagnostics = adversarial_loss(
        batch,
        {"xy": PairCritic(), "xz_yz": PairCritic()},
        {"xy": (0,), "xz_yz": (1, 2)},
        diagnose=diagnose,
    )
    expected = sum(
        F.softplus(-(weight * factors[axis] + 0.25)) / divisor
        for axis, divisor in ((0, 2), (1, 4), (2, 4))
    )
    assert weight.grad is None
    assert current.requires_grad is False
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert len(diagnostics) == (6 if diagnose else 0)
    assert all(not value.requires_grad for value in diagnostics.values())
    assert all(value > 0 for value in diagnostics.values())


def test_empty_adversarial_batch_returns_differentiable_zero():
    logits = torch.ones(1, 2, 3, 3, 3, requires_grad=True)
    empty = torch.empty(0, 2, 3, 3)
    batch = SimpleNamespace(fake={0: (empty, empty)}, logits=logits)
    loss, diagnostics = adversarial_loss(batch, {}, {"xy": (0,)}, True)
    assert loss == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert diagnostics == {}


def test_profile_loss_supervises_height_only_samples():
    # Uniform fake against a real profile that rises with depth; CFG dropped vf.
    probs = torch.full((2, 2, 4, 2, 2), 0.5, requires_grad=True)
    rising = torch.tensor([0.0, 0.1, 0.2, 0.3])
    profile = torch.stack((1 - rising, rising)).expand(2, 2, 4)
    batch = SimpleNamespace(
        clean_probs=probs, profile=profile, vf_present=torch.tensor([True, False])
    )
    settings = SimpleNamespace(
        profile_bins=4,
        profile_gradient_weight=0.0,
        profile_weight=1.0,
        num_phases=2,
    )
    loss, logs = profile_loss(batch, settings)
    loss.backward()
    assert probs.grad[1].abs().sum() > 0
    assert probs.grad[0].abs().sum() > 0
    assert logs["profile/soft_mae"] > 0
