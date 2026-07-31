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
        global_loss=F.softplus(-real_scores.global_logits).mean()
        + F.softplus(fake_scores.global_logits).mean(),
        local_loss=F.softplus(-real_scores.local_logits).mean()
        + F.softplus(fake_scores.local_logits).mean(),
    )


def generator_logistic_loss(fake_scores: CriticScores) -> HeadLoss:
    return HeadLoss(
        global_loss=F.softplus(-fake_scores.global_logits).mean(),
        local_loss=F.softplus(-fake_scores.local_logits).mean(),
    )


def aggregate_r1_scores(
    scores: CriticScores,
    local_weight: float,
) -> torch.Tensor:
    return scores.global_logits + local_weight * scores.local_logits.mean(dim=(-2, -1))


def r1_penalty(
    real_logits: torch.Tensor,
    real_inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=tuple(real_inputs),
        create_graph=True,
        only_inputs=True,
    )
    batch = real_logits.shape[0]
    squared_norm = torch.zeros(batch, device=real_logits.device)
    for gradient in gradients:
        squared_norm = squared_norm + gradient.square().reshape(batch, -1).sum(1)
    return squared_norm.mean()
