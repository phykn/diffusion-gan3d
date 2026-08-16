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
    groups: torch.Tensor | None = None,
) -> HeadLoss:
    if groups is None:
        return HeadLoss(
            global_loss=(
                F.softplus(-real_scores.logits_global).mean()
                + F.softplus(fake_scores.logits_global).mean()
            ),
            local_loss=(
                F.softplus(-real_scores.logits_local).mean()
                + F.softplus(fake_scores.logits_local).mean()
            ),
        )
    return HeadLoss(
        global_loss=_group_mean(
            F.softplus(-real_scores.logits_global)
            + F.softplus(fake_scores.logits_global),
            groups,
        ),
        local_loss=_group_mean(
            F.softplus(-real_scores.logits_local)
            + F.softplus(fake_scores.logits_local),
            groups,
        ),
    )


def get_generator_loss(
    fake_scores: CriticScores,
    groups: torch.Tensor | None = None,
) -> HeadLoss:
    return HeadLoss(
        global_loss=_group_mean(F.softplus(-fake_scores.logits_global), groups),
        local_loss=_group_mean(F.softplus(-fake_scores.logits_local), groups),
    )


def _group_mean(values: torch.Tensor, groups: torch.Tensor | None) -> torch.Tensor:
    if values.ndim == 0:
        raise ValueError("loss values must include a batch dimension.")
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    if groups is None:
        return per_sample.mean()
    if groups.shape != (values.shape[0],) or groups.dtype != torch.bool:
        raise ValueError("loss groups must be boolean with shape [B].")
    means = [
        per_sample[groups == group].mean()
        for group in (False, True)
        if bool((groups == group).any())
    ]
    return torch.stack(means).mean()


def get_critic_r1(
    scores: CriticScores,
    real_inputs: Sequence[torch.Tensor],
    groups: torch.Tensor | None = None,
) -> HeadLoss:
    return HeadLoss(
        global_loss=get_r1(scores.logits_global, real_inputs, groups),
        local_loss=get_r1(
            scores.logits_local.mean(dim=(-2, -1)),
            real_inputs,
            groups,
        ),
    )


def get_r1(
    logits: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    groups: torch.Tensor | None = None,
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
    return _group_mean(norm, groups)
