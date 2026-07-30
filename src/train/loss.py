from collections.abc import Sequence

import torch
import torch.nn.functional as F


def critic_logistic_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def generator_logistic_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fake_logits).mean()


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
