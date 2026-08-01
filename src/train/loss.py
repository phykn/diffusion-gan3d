from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model.critic import CriticScores


@dataclass(frozen=True)
class HeadLoss:
    global_loss: torch.Tensor
    local_loss: torch.Tensor

    def combine(self, local_weight: float) -> torch.Tensor:
        return self.global_loss + local_weight * self.local_loss


def get_critic_loss(
    real_scores: CriticScores,
    fake_scores: CriticScores,
) -> HeadLoss:
    return HeadLoss(
        global_loss=F.softplus(-real_scores.logits_global).mean()
        + F.softplus(fake_scores.logits_global).mean(),
        local_loss=F.softplus(-real_scores.logits_local).mean()
        + F.softplus(fake_scores.logits_local).mean(),
    )


def get_generator_loss(fake_scores: CriticScores) -> HeadLoss:
    return HeadLoss(
        global_loss=F.softplus(-fake_scores.logits_global).mean(),
        local_loss=F.softplus(-fake_scores.logits_local).mean(),
    )


def get_critic_r1(
    scores: CriticScores,
    real_inputs: Sequence[torch.Tensor],
) -> HeadLoss:
    return HeadLoss(
        global_loss=get_r1(scores.logits_global, real_inputs),
        local_loss=get_r1(
            scores.logits_local.mean(dim=(-2, -1)),
            real_inputs,
        ),
    )


def get_r1(
    logits: torch.Tensor,
    inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=tuple(inputs),
        create_graph=True,
        only_inputs=True,
    )
    batch = logits.shape[0]
    norm = torch.zeros(batch, device=logits.device)
    for grad in grads:
        norm = norm + grad.square().reshape(batch, -1).sum(1)
    return norm.mean()
