import torch.nn.functional as F

from src.prepare.profile import rebin_profile


def compute_profile_loss(probs, target, present, bins, profile_weight=1.0):
    predicted = rebin_profile(probs.float().mean(dim=(-1, -2)), bins)
    target = rebin_profile(target, bins).to(predicted)
    loss = profile_weight * F.smooth_l1_loss(predicted, target, reduction="none").mean(
        (1, 2)
    )
    mask = present.to(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1)
