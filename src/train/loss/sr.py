import torch

from src.prepare.resize import downsample


def consistency_loss(
    high: torch.Tensor,
    low: torch.Tensor,
    tolerance: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructed = downsample(high, tuple(low.shape[2:]))
    squared = (reconstructed - low).square()
    if mask is None:
        error = squared.mean()
    else:
        mask = mask.to(squared)
        error = (squared * mask).sum() / (mask.sum() * low.shape[1]).clamp_min(1)
    # A dead zone lets fine boundaries adapt without forcing block-shaped phases.
    return (error - tolerance).clamp_min(0), error
