import torch

from src.prepare.resize import downsample


def consistency_loss(
    high: torch.Tensor, low: torch.Tensor, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructed = downsample(high, tuple(low.shape[2:]))
    error = (reconstructed - low).square().mean()
    # A dead zone lets fine boundaries adapt without forcing block-shaped phases.
    return (error - tolerance).clamp_min(0), error
