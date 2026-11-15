from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CriticObjective:
    adversarial: torch.Tensor
    total: torch.Tensor
    r1: torch.Tensor
    r2: torch.Tensor


def relativistic_loss(preferred, other):
    """RpGAN loss: each other sample should score below its paired preferred one.

    Pairs remove the logistic loss's mode-dropping optima; the smaller batch is
    reused to pair every sample.
    """
    losses = []
    for first, second in zip(preferred, other, strict=True):
        index = torch.arange(max(len(first), len(second)), device=first.device)
        margin = second[index % len(second)] - first[index % len(first)]
        losses.append(F.softplus(margin).mean())
    return torch.stack(losses).mean()


def critic_objective(
    adversarial,
    real_scores,
    fake_scores,
    real_inputs,
    fake_inputs,
    r1_weight,
    r2_weight,
    interval,
):
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


def gradient_penalty(scores, inputs):
    """Mean squared input-gradient norm of each score scale."""
    penalties = []
    for logits in scores:
        grads = torch.autograd.grad(logits.sum(), inputs, create_graph=True)
        batch = logits.shape[0]
        norm = torch.zeros(batch, device=logits.device)
        for grad in grads:
            norm = norm + grad.square().reshape(batch, -1).sum(1)
        penalties.append(norm.mean())
    return torch.stack(penalties).mean()
