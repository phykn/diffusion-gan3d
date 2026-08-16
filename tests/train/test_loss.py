import torch
import torch.nn.functional as F

from src.model.critic import CriticScores
from src.train.loss import (
    get_critic_loss,
    get_critic_r1,
    get_generator_loss,
)


def test_logistic_losses_average_each_head_before_weighting() -> None:
    real = CriticScores(
        logits_global=torch.tensor([-1.0, 2.0]),
        logits_local=torch.tensor(
            [
                [[-2.0, 0.0], [1.0, 3.0]],
                [[-1.0, 2.0], [0.5, 1.5]],
            ]
        ),
    )
    fake = CriticScores(
        logits_global=torch.tensor([0.5, -0.5]),
        logits_local=torch.tensor(
            [
                [[1.0, -1.0], [2.0, 0.0]],
                [[-2.0, 0.5], [1.5, -0.5]],
            ]
        ),
    )

    critic = get_critic_loss(real, fake)
    generator = get_generator_loss(fake)

    want_d_global = (
        F.softplus(-real.logits_global).mean() + F.softplus(fake.logits_global).mean()
    )
    want_d_local = (
        F.softplus(-real.logits_local).mean() + F.softplus(fake.logits_local).mean()
    )
    want_g_global = F.softplus(-fake.logits_global).mean()
    want_g_local = F.softplus(-fake.logits_local).mean()
    assert torch.allclose(critic.global_loss, want_d_global)
    assert torch.allclose(critic.local_loss, want_d_local)
    assert torch.allclose(
        critic.combine(0.5),
        want_d_global + 0.5 * want_d_local,
    )
    assert torch.allclose(generator.global_loss, want_g_global)
    assert torch.allclose(generator.local_loss, want_g_local)
    assert torch.allclose(
        generator.combine(0.5),
        want_g_global + 0.5 * want_g_local,
    )


def test_critic_loss_allows_different_real_and_fake_batch_sizes() -> None:
    real = CriticScores(
        logits_global=torch.linspace(-1.0, 1.0, 8),
        logits_local=torch.linspace(-2.0, 2.0, 32).reshape(8, 2, 2),
    )
    fake = CriticScores(
        logits_global=torch.linspace(-1.5, 1.5, 16),
        logits_local=torch.linspace(-3.0, 3.0, 64).reshape(16, 2, 2),
    )

    loss = get_critic_loss(real, fake)

    assert torch.allclose(
        loss.global_loss,
        F.softplus(-real.logits_global).mean()
        + F.softplus(fake.logits_global).mean(),
    )
    assert torch.allclose(
        loss.local_loss,
        F.softplus(-real.logits_local).mean()
        + F.softplus(fake.logits_local).mean(),
    )


def test_r1_aggregation_is_independent_of_patch_count() -> None:
    small = _r1_penalty(2)
    large = _r1_penalty(16)

    assert torch.allclose(small, large)
    assert torch.allclose(small, torch.tensor(8.5))


def test_connectivity_groups_receive_equal_weight_despite_unequal_counts() -> None:
    logits = torch.tensor((0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0))
    scores = CriticScores(
        logits_global=logits,
        logits_local=logits[:, None, None],
    )
    groups = torch.tensor((True, False, False, False, False, False, False))

    loss = get_generator_loss(scores, groups)
    expected = 0.5 * (F.softplus(-logits[0]) + F.softplus(-logits[1]))

    assert torch.allclose(loss.global_loss, expected)
    assert torch.allclose(loss.local_loss, expected)


def test_r1_heads_do_not_cancel_opposite_gradients() -> None:
    inputs = torch.tensor([[1.0], [2.0]], requires_grad=True)
    base = inputs[:, 0]
    scores = CriticScores(
        logits_global=2.0 * base,
        logits_local=(-4.0 * base[:, None, None]).expand(-1, 2, 2),
    )

    penalties = get_critic_r1(scores, (inputs,))

    assert torch.allclose(penalties.global_loss, torch.tensor(4.0))
    assert torch.allclose(penalties.local_loss, torch.tensor(16.0))
    assert torch.allclose(penalties.combine(0.5), torch.tensor(12.0))


def _r1_penalty(size: int) -> torch.Tensor:
    inputs = torch.tensor([[1.0], [2.0]], requires_grad=True)
    base = inputs[:, 0]
    scores = CriticScores(
        logits_global=2.0 * base,
        logits_local=(3.0 * base[:, None, None]).expand(-1, size, size),
    )
    return get_critic_r1(scores, (inputs,)).combine(0.5)
