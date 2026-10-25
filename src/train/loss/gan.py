from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.model.critic import CriticScores


@dataclass(frozen=True)
class HeadLoss:
    global_loss: torch.Tensor
    local_loss: torch.Tensor | None = None

    def combine(self, local_weight: float) -> torch.Tensor:
        if self.local_loss is None:
            return self.global_loss
        return self.global_loss + local_weight * self.local_loss


@dataclass(frozen=True)
class CriticObjective:
    adversarial: HeadLoss
    total: torch.Tensor
    r1: torch.Tensor
    r2: torch.Tensor


def score_plane(critic, previous, current, time, domain, **conditions):
    if getattr(critic, "input_mode", "pair") == "single":
        return critic(previous, time, domain, **conditions)
    return critic(previous, current, time, domain, **conditions)


def score_triplets(critic, batch, values, domains):
    return critic(
        values,
        batch.axes,
        batch.gaps,
        domains,
        **({"height": batch.height} if batch.height is not None else {}),
        **({"profile": batch.profile} if batch.profile is not None else {}),
    )


def active_groups(fake, groups):
    active = {
        group: tuple(axis for axis in axes if len(fake[axis][0]))
        for group, axes in groups.items()
    }
    return {group: axes for group, axes in active.items() if axes}


def get_critic_loss(
    real_scores: CriticScores,
    fake_scores: CriticScores,
) -> HeadLoss:
    if real_scores.levels or fake_scores.levels:
        return mean_heads(
            [
                get_critic_loss(real, fake)
                for real, fake in zip(
                    real_scores.levels or (real_scores,),
                    fake_scores.levels or (fake_scores,),
                    strict=True,
                )
            ]
        )
    return HeadLoss(
        global_loss=(
            F.softplus(-real_scores.logits_global).mean()
            + F.softplus(fake_scores.logits_global).mean()
        ),
        local_loss=(
            F.softplus(-real_scores.logits_local).mean()
            + F.softplus(fake_scores.logits_local).mean()
        )
        if real_scores.logits_local is not None
        else None,
    )


def get_relativistic_loss(
    preferred: CriticScores,
    other: CriticScores,
) -> HeadLoss:
    """RpGAN loss: each other sample should score below its paired preferred one.

    Judging pairs instead of single samples removes the mode-dropping optima of
    the logistic loss, where a generator covering one mode of the data already
    looks real. The smaller batch is reused to pair every sample.
    """
    if preferred.levels or other.levels:
        return mean_heads(
            [
                get_relativistic_loss(first, second)
                for first, second in zip(
                    preferred.levels or (preferred,),
                    other.levels or (other,),
                    strict=True,
                )
            ]
        )
    first, second = preferred.logits_global, other.logits_global
    index = torch.arange(max(len(first), len(second)), device=first.device)
    margin = second[index % len(second)] - first[index % len(first)]
    return HeadLoss(global_loss=F.softplus(margin).mean())


def critic_objective(
    adversarial: HeadLoss,
    real_scores: CriticScores,
    fake_scores: CriticScores,
    real_inputs: Sequence[torch.Tensor],
    fake_inputs: Sequence[torch.Tensor],
    r1_weight: float,
    r2_weight: float,
    interval: int,
    local_weight: float = 0.0,
) -> CriticObjective:
    total = adversarial.combine(local_weight)
    r1 = total.new_zeros(())
    r2 = total.new_zeros(())
    if r1_weight > 0:
        r1 = get_gradient_penalty(real_scores, real_inputs).combine(local_weight)
        total = total + 0.5 * r1_weight * interval * r1
    if r2_weight > 0:
        r2 = get_gradient_penalty(fake_scores, fake_inputs).combine(local_weight)
        total = total + 0.5 * r2_weight * interval * r2
    return CriticObjective(adversarial, total, r1, r2)


def get_generator_loss(
    fake_scores: CriticScores,
) -> HeadLoss:
    if fake_scores.levels:
        return mean_heads([get_generator_loss(level) for level in fake_scores.levels])
    return HeadLoss(
        global_loss=F.softplus(-fake_scores.logits_global).mean(),
        local_loss=(
            F.softplus(-fake_scores.logits_local).mean()
            if fake_scores.logits_local is not None
            else None
        ),
    )


def get_gradient_penalty(
    scores: CriticScores,
    inputs: Sequence[torch.Tensor],
) -> HeadLoss:
    if scores.levels:
        return mean_heads(
            [get_gradient_penalty(level, inputs) for level in scores.levels]
        )
    return HeadLoss(
        global_loss=input_gradient_penalty(scores.logits_global, inputs),
        local_loss=input_gradient_penalty(
            scores.logits_local.mean(dim=(-2, -1)),
            inputs,
        )
        if scores.logits_local is not None
        else None,
    )


def mean_heads(heads):
    return HeadLoss(
        torch.stack([head.global_loss for head in heads]).mean(),
        torch.stack([head.local_loss for head in heads]).mean()
        if heads[0].local_loss is not None
        else None,
    )


def input_gradient_penalty(
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
