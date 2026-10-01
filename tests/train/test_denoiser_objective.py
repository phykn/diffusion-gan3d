from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.model.critic import CriticScores
from src.train.loss.denoiser import adversarial_loss


class PairCritic(nn.Module):
    def forward(self, previous, current, time, domain):
        score = previous.flatten(1).mean(1) + 0.25 * current.flatten(1).mean(1)
        return CriticScores(score, 2 * score[:, None, None])


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
    head, diagnostics = adversarial_loss(
        batch,
        {"xy": PairCritic(), "xz_yz": PairCritic()},
        {"xy": (0,), "xz_yz": (1, 2)},
        local_weight=0.3,
        diagnose=diagnose,
    )
    expected = sum(
        (
            F.softplus(-(weight * factors[axis] + 0.25))
            + 0.3 * F.softplus(-2 * (weight * factors[axis] + 0.25))
        )
        / divisor
        for axis, divisor in ((0, 2), (1, 4), (2, 4))
    )
    assert weight.grad is None
    assert current.requires_grad is False
    torch.testing.assert_close(head.combine(0.3), expected)
    actual_grad = torch.autograd.grad(head.combine(0.3), weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert len(diagnostics) == (6 if diagnose else 0)
    assert all(not value.requires_grad for value in diagnostics.values())
    assert all(value > 0 for value in diagnostics.values())


def test_empty_adversarial_batch_returns_differentiable_zero():
    logits = torch.ones(1, 2, 3, 3, 3, requires_grad=True)
    empty = torch.empty(0, 2, 3, 3)
    batch = SimpleNamespace(fake={0: (empty, empty)}, logits=logits)
    head, diagnostics = adversarial_loss(batch, {}, {"xy": (0,)}, 1, True)
    assert head.combine(1) == 0
    head.combine(1).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert diagnostics == {}
