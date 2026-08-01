from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model.critic import CriticScores


@dataclass(frozen=True)
class HeadLoss:
    global_loss: torch.Tensor
    local_loss: torch.Tensor

    def total(self, local_weight: float) -> torch.Tensor:
        return self.global_loss + local_weight * self.local_loss


def critic_logistic_loss(
    real_scores: CriticScores,
    fake_scores: CriticScores,
) -> HeadLoss:
    return HeadLoss(
        global_loss=F.softplus(-real_scores.logits_global).mean()
        + F.softplus(fake_scores.logits_global).mean(),
        local_loss=F.softplus(-real_scores.logits_local).mean()
        + F.softplus(fake_scores.logits_local).mean(),
    )


def generator_logistic_loss(fake_scores: CriticScores) -> HeadLoss:
    return HeadLoss(
        global_loss=F.softplus(-fake_scores.logits_global).mean(),
        local_loss=F.softplus(-fake_scores.logits_local).mean(),
    )


def critic_r1_penalty(
    scores: CriticScores,
    real_inputs: Sequence[torch.Tensor],
) -> HeadLoss:
    return HeadLoss(
        global_loss=r1_penalty(scores.logits_global, real_inputs),
        local_loss=r1_penalty(
            scores.logits_local.mean(dim=(-2, -1)),
            real_inputs,
        ),
    )


def r1_penalty(
    real_logits: torch.Tensor,
    real_inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=tuple(real_inputs),
        create_graph=True,
        only_inputs=True,
    )
    batch = real_logits.shape[0]
    norm = torch.zeros(batch, device=real_logits.device)
    for grad in grads:
        norm = norm + grad.square().reshape(batch, -1).sum(1)
    return norm.mean()
