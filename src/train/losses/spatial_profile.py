import torch.nn.functional as F

from src.data.height import rebin_profile


def profile_loss(probs, target, bins, weight=1.0):
    predicted = rebin_profile(probs.float().mean(dim=(-1, -2)), bins)
    target = rebin_profile(target, bins).to(predicted)
    loss = weight * F.smooth_l1_loss(predicted, target, reduction="none").mean((1, 2))
    return loss.mean()
