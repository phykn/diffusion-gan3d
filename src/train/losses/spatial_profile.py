"""Depth profile of the predicted volume against the real crop profile."""

import torch.nn.functional as F

from src.data.height import rebin_profile


def profile_loss(probs, target, present, bins, weight=1.0):
    predicted = rebin_profile(probs.float().mean(dim=(-1, -2)), bins)
    target = rebin_profile(target, bins).to(predicted)
    loss = weight * F.smooth_l1_loss(predicted, target, reduction="none").mean((1, 2))
    mask = present.to(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1)
