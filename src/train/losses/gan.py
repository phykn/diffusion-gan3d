from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.model.critic import CriticScores


@dataclass(frozen=True)
class CriticObjective:
    adversarial: torch.Tensor
    total: torch.Tensor
    r1: torch.Tensor
    r2: torch.Tensor


def relativistic_loss(preferred: CriticScores, other: CriticScores) -> torch.Tensor:
    """RpGAN loss: each other sample should score below its paired preferred one.

    Judging pairs instead of single samples removes the mode-dropping optima of
    the logistic loss, where a generator covering one mode of the data already
    looks real. The smaller batch is reused to pair every sample.
    """
    if preferred.levels or other.levels:
        return torch.stack(
            [
                relativistic_loss(first, second)
                for first, second in zip(
                    preferred.levels or (preferred,),
                    other.levels or (other,),
                    strict=True,
                )
            ]
        ).mean()
    first, second = preferred.logits_global, other.logits_global
    index = torch.arange(max(len(first), len(second)), device=first.device)
    margin = second[index % len(second)] - first[index % len(first)]
    return F.softplus(margin).mean()


def critic_objective(
    adversarial: torch.Tensor,
    real_scores: CriticScores,
    fake_scores: CriticScores,
    real_inputs: Sequence[torch.Tensor],
    fake_inputs: Sequence[torch.Tensor],
    r1_weight: float,
    r2_weight: float,
    interval: int,
) -> CriticObjective:
    """Lazy R1 (reals) and R2 (fakes), scaled by the regularization interval."""
    total = adversarial
    r1 = total.new_zeros(())
    r2 = total.new_zeros(())
    if r1_weight > 0:
        r1 = gradient_penalty(real_scores, real_inputs)
        total = total + 0.5 * r1_weight * interval * r1
    if r2_weight > 0:
        r2 = gradient_penalty(fake_scores, fake_inputs)
        total = total + 0.5 * r2_weight * interval * r2
    return CriticObjective(adversarial, total, r1, r2)


def gradient_penalty(
    scores: CriticScores, inputs: Sequence[torch.Tensor]
) -> torch.Tensor:
    if scores.levels:
        return torch.stack(
            [gradient_penalty(level, inputs) for level in scores.levels]
        ).mean()
    return input_gradient_penalty(scores.logits_global, inputs)


def input_gradient_penalty(
    logits: torch.Tensor, inputs: Sequence[torch.Tensor]
) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs=logits.sum(), inputs=tuple(inputs), create_graph=True, only_inputs=True
    )
    batch = logits.shape[0]
    norm = torch.zeros(batch, device=logits.device)
    for grad in grads:
        norm = norm + grad.square().reshape(batch, -1).sum(1)
    return norm.mean()
