import torch.nn.functional as F

from src.prepare.profile import rebin_profile


def compute_profile_loss(
    probs, target, present, bins, gradient_weight=0.0, profile_weight=1.0
):
    predicted = rebin_profile(probs.float().mean(dim=(-1, -2)), bins)
    target = rebin_profile(target, bins).to(predicted)
    loss = profile_weight * F.smooth_l1_loss(predicted, target, reduction="none").mean(
        (1, 2)
    )
    if gradient_weight and bins > 1:
        loss = loss + gradient_weight * F.smooth_l1_loss(
            predicted.diff(), target.diff(), reduction="none"
        ).mean((1, 2))
    mask = present.to(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1)
