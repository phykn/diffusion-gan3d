import torch
import torch.nn.functional as F

from src.model.critic import CriticScores
from src.train.loss import (
    critic_logistic_loss,
    critic_r1_penalty,
    generator_logistic_loss,
)


def test_logistic_losses_average_each_head_before_weighting() -> None:
    real = CriticScores(
        global_logits=torch.tensor([-1.0, 2.0]),
        local_logits=torch.tensor(
            [
                [[-2.0, 0.0], [1.0, 3.0]],
                [[-1.0, 2.0], [0.5, 1.5]],
            ]
        ),
    )
    fake = CriticScores(
        global_logits=torch.tensor([0.5, -0.5]),
        local_logits=torch.tensor(
            [
                [[1.0, -1.0], [2.0, 0.0]],
                [[-2.0, 0.5], [1.5, -0.5]],
            ]
        ),
    )

    critic = critic_logistic_loss(real, fake)
    generator = generator_logistic_loss(fake)

    expected_critic_global = (
        F.softplus(-real.global_logits).mean() + F.softplus(fake.global_logits).mean()
    )
    expected_critic_local = (
        F.softplus(-real.local_logits).mean() + F.softplus(fake.local_logits).mean()
    )
    expected_generator_global = F.softplus(-fake.global_logits).mean()
    expected_generator_local = F.softplus(-fake.local_logits).mean()
    assert torch.allclose(critic.global_loss, expected_critic_global)
    assert torch.allclose(critic.local_loss, expected_critic_local)
    assert torch.allclose(
        critic.total(0.5),
        expected_critic_global + 0.5 * expected_critic_local,
    )
    assert torch.allclose(generator.global_loss, expected_generator_global)
    assert torch.allclose(generator.local_loss, expected_generator_local)
    assert torch.allclose(
        generator.total(0.5),
        expected_generator_global + 0.5 * expected_generator_local,
    )


def test_r1_aggregation_is_independent_of_patch_count() -> None:
    small = _r1_penalty(2)
    large = _r1_penalty(16)

    assert torch.allclose(small, large)
    assert torch.allclose(small, torch.tensor(8.5))


def test_r1_heads_do_not_cancel_opposite_gradients() -> None:
    inputs = torch.tensor([[1.0], [2.0]], requires_grad=True)
    base = inputs[:, 0]
    scores = CriticScores(
        global_logits=2.0 * base,
        local_logits=(-4.0 * base[:, None, None]).expand(-1, 2, 2),
    )

    penalties = critic_r1_penalty(scores, (inputs,))

    assert torch.allclose(penalties.global_loss, torch.tensor(4.0))
    assert torch.allclose(penalties.local_loss, torch.tensor(16.0))
    assert torch.allclose(penalties.total(0.5), torch.tensor(12.0))


def _r1_penalty(size: int) -> torch.Tensor:
    inputs = torch.tensor([[1.0], [2.0]], requires_grad=True)
    base = inputs[:, 0]
    scores = CriticScores(
        global_logits=2.0 * base,
        local_logits=(3.0 * base[:, None, None]).expand(-1, size, size),
    )
    return critic_r1_penalty(scores, (inputs,)).total(0.5)
