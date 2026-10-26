import pytest
import torch
import torch.nn.functional as F

from src.model.critic import CriticScores, PairCritic2D
from src.train.loss.gan import (
    critic_objective,
    get_gradient_penalty,
    get_relativistic_loss,
)


def test_relativistic_loss_pairs_samples_and_reuses_the_smaller_batch() -> None:
    real = CriticScores(torch.tensor([1.0, -2.0]))
    fake = CriticScores(torch.tensor([0.5, 0.0, 3.0]))

    critic = get_relativistic_loss(real, fake)
    generator = get_relativistic_loss(fake, real)

    margins = torch.tensor([0.5 - 1.0, 0.0 + 2.0, 3.0 - 1.0])
    torch.testing.assert_close(critic.global_loss, F.softplus(margins).mean())
    torch.testing.assert_close(generator.global_loss, F.softplus(-margins).mean())


def test_relativistic_loss_depends_only_on_score_differences() -> None:
    # An offset shared by both sides of every pair cannot change the loss.
    real = CriticScores(torch.tensor([0.3, -0.4]))
    fake = CriticScores(torch.tensor([-0.2, 0.1]))
    shifted_real = CriticScores(real.logits_global + 5.0)
    shifted_fake = CriticScores(fake.logits_global + 5.0)

    torch.testing.assert_close(
        get_relativistic_loss(shifted_real, shifted_fake).global_loss,
        get_relativistic_loss(real, fake).global_loss,
    )


def test_pyramid_averages_losses_after_nonlinearity():
    critic = PairCritic2D(2, (4, 8), 8, 1, pyramid_min_size=4)
    previous = torch.randn(2, 2, 16, 16, requires_grad=True)
    current = torch.randn_like(previous)
    scores = critic(
        previous,
        current,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    assert len(scores.levels) == 3
    zero = CriticScores(
        torch.zeros(2), tuple(CriticScores(torch.zeros(2)) for _ in scores.levels)
    )
    loss = get_relativistic_loss(scores, zero)
    expected = torch.stack(
        [F.softplus(-s.logits_global).mean() for s in scores.levels]
    ).mean()
    torch.testing.assert_close(loss.global_loss, expected)
    loss.global_loss.backward()
    assert torch.isfinite(previous.grad).all()


@pytest.mark.parametrize("levels", [1, 3])
@pytest.mark.parametrize(
    "r1_weight,r2_weight", [(0, 0), (0.3, 0), (0, 0.2), (0.3, 0.2)]
)
def test_objective_and_penalties_preserve_gradients(levels, r1_weight, r2_weight):
    real = torch.tensor([[1.0], [2.0]], requires_grad=True)
    fake = torch.tensor([[-1.0], [0.0], [1.0]], requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor(2.0))
    real_levels = tuple(
        CriticScores(k * real[:, 0] * weight) for k in range(1, levels + 1)
    )
    fake_levels = tuple(
        CriticScores(2 * k * fake[:, 0] * weight) for k in range(1, levels + 1)
    )
    real_scores = CriticScores(real_levels[0].logits_global, levels=real_levels)
    fake_scores = CriticScores(fake_levels[0].logits_global, levels=fake_levels)
    result = critic_objective(
        get_relativistic_loss(real_scores, fake_scores),
        real_scores,
        fake_scores,
        (real,),
        (fake,),
        r1_weight,
        r2_weight,
        4,
    )
    penalty = sum(k**2 for k in range(1, levels + 1)) / levels * weight.square()
    expected_r1 = penalty if r1_weight else weight.new_zeros(())
    expected_r2 = 4 * penalty if r2_weight else weight.new_zeros(())
    # Three fakes pair with reals [0, 1, 0].
    adversarial = torch.stack(
        [
            F.softplus(f.logits_global - r.logits_global[[0, 1, 0]]).mean()
            for r, f in zip(real_levels, fake_levels, strict=True)
        ]
    ).mean()
    expected = adversarial + 2 * r1_weight * expected_r1 + 2 * r2_weight * expected_r2
    expected_grad = torch.autograd.grad(expected, weight, retain_graph=True)[0]
    torch.testing.assert_close(
        get_gradient_penalty(real_scores, (real,)).global_loss, penalty
    )
    torch.testing.assert_close(result.total, expected)
    torch.testing.assert_close(result.r1, expected_r1)
    torch.testing.assert_close(result.r2, expected_r2)
    result.total.backward()
    torch.testing.assert_close(weight.grad, expected_grad)
