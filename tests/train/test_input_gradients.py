import pytest
import torch

from src.model.critic import CriticScores
from src.train.loss.gan import get_generator_loss
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
    scores = CriticScores(logits, logits[:, None, None])
    loss = get_generator_loss(scores).combine(0.5)
    norms = input_gradient_norms(loss * len(previous), (previous, current))
    active = 0 if source == "previous" else 1
    assert norms[active] > 0 and norms[1 - active] == 0
    assert all(not norm.requires_grad for norm in norms)
    assert previous.grad is current.grad is weight.grad is None
    loss.backward()
    assert selected.grad is not None and selected.grad.abs().sum() > 0
    assert weight.grad is not None


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("local_weight", [0.0, 0.6])
def test_diagnostics_include_local_pyramid_objective_without_batch_size_bias(
    batch, local_weight
):
    previous = torch.zeros(batch, 1, 2, 2, requires_grad=True)
    current = torch.zeros_like(previous, requires_grad=True)
    levels = tuple(
        CriticScores(torch.zeros(batch), factor * previous[:, 0]) for factor in (1, 3)
    )
    scores = CriticScores(torch.zeros(batch), torch.zeros(batch, 2, 2), levels)
    loss = get_generator_loss(scores).combine(local_weight)
    prev_norm, curr_norm = input_gradient_norms(loss * batch, (previous, current))
    # At zero, softplus(-x)' = -1/2; mean level slope is 2, across 4 pixels.
    assert prev_norm.item() == pytest.approx(0.5 * local_weight)
    assert curr_norm == 0


def test_constant_score_has_zero_input_sensitivity():
    inputs = (torch.zeros(2, 1, 2, 2, requires_grad=True), torch.zeros(2, 1, 2, 2))
    assert all(value == 0 for value in input_gradient_norms(torch.tensor(1.0), inputs))
