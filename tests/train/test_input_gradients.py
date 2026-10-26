import pytest
import torch

from src.model.critic import CriticScores
from src.train.loss.gan import get_relativistic_loss
from src.train.step import input_gradient_norms


@pytest.mark.parametrize("source", ["previous", "current"])
def test_input_diagnostics_distinguish_condition_only_critics_without_accumulation(
    source,
):
    previous = torch.zeros(2, 3, 4, 4, requires_grad=True)
    current = torch.zeros_like(previous, requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor(2.0))
    selected = previous if source == "previous" else current
    logits = selected.mean((1, 2, 3)) * weight
    loss = get_relativistic_loss(CriticScores(logits), CriticScores(torch.zeros(2)))
    loss = loss.global_loss
    norms = input_gradient_norms(loss * len(previous), (previous, current))
    active = 0 if source == "previous" else 1
    assert norms[active] > 0 and norms[1 - active] == 0
    assert all(not norm.requires_grad for norm in norms)
    assert previous.grad is current.grad is weight.grad is None
    loss.backward()
    assert selected.grad is not None and selected.grad.abs().sum() > 0
    assert weight.grad is not None


@pytest.mark.parametrize("batch", [1, 4])
def test_pyramid_diagnostics_have_no_batch_size_bias(batch):
    previous = torch.zeros(batch, 1, 2, 2, requires_grad=True)
    current = torch.zeros_like(previous, requires_grad=True)
    levels = tuple(CriticScores(factor * previous.sum((1, 2, 3))) for factor in (1, 3))
    zero = CriticScores(
        torch.zeros(batch), tuple(CriticScores(torch.zeros(batch)) for _ in levels)
    )
    loss = get_relativistic_loss(CriticScores(levels[0].logits_global, levels), zero)
    prev_norm, curr_norm = input_gradient_norms(
        loss.global_loss * batch, (previous, current)
    )
    # At zero, softplus(-x)' = -1/2; the mean level slope of 2 reaches 4 pixels.
    assert prev_norm.item() == pytest.approx(2.0)
    assert curr_norm == 0


def test_constant_score_has_zero_input_sensitivity():
    inputs = (torch.zeros(2, 1, 2, 2, requires_grad=True), torch.zeros(2, 1, 2, 2))
    assert all(value == 0 for value in input_gradient_norms(torch.tensor(1.0), inputs))
