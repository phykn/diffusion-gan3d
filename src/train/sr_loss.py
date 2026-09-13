import torch
from torch import nn

from src.prepare.resize import downsample


def consistency_loss(
    high: torch.Tensor, low: torch.Tensor, temperature: float, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructed = downsample(high, tuple(low.shape[2:]), temperature)
    error = (reconstructed - low).square().mean()
    # A dead zone lets fine boundaries adapt without forcing block-shaped phases.
    return (error - tolerance).clamp_min(0), error


def gradient_penalty(
    critic: nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    mixed = torch.lerp(real.float(), fake.detach().float(), alpha).requires_grad_(True)
    scores = critic(mixed)
    (gradient,) = torch.autograd.grad(scores.sum(), mixed, create_graph=True)
    return (gradient.flatten(1).norm(2, dim=1) - 1).square().mean()


def sample_slices(volume: torch.Tensor, axis: int, count: int) -> torch.Tensor:
    # Move the slicing axis next to batch: B,L,K,H,W.
    planes = volume.movedim(axis + 2, 1)
    planes = planes.flatten(0, 1)
    indices = torch.randint(planes.shape[0], (count,), device=volume.device)
    return planes[indices]
