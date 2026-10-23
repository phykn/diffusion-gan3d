import torch

from src.anchor import AnchorCondition

# Allowed change-rate deviation, in median absolute deviations. In unanchored LR
# samples about 1% of adjacent-plane pairs exceed it; most seam pairs next to an
# anchored block do.
OUTLIER_SCALE = 4.0


def compute_anchor_neighborhood_loss(
    probs: torch.Tensor,
    condition: AnchorCondition,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep every adjacent-plane change rate on an anchored axis near its median.

    Each distance from the nearest anchor is a separate term. A window that stops
    at some radius leaves the next pair free, and the generator moves the seam there.
    """
    zero = probs.sum(dtype=torch.float32) * 0.0
    length = max(probs.shape[2:])
    totals = [probs.new_zeros(length, dtype=torch.float32) for _ in range(4)]
    reference_axes = zero.detach()
    with torch.autocast(device_type=probs.device.type, enabled=False):
        for batch in condition.active_batches:
            for axis in range(3):
                regions = [
                    region for region in condition.regions if region.axis == axis
                ]
                if not regions:
                    continue
                values = probs[batch].movedim(axis + 1, 1)
                fixed = condition.mask[batch, 0].movedim(axis, 0)
                observed = condition.image[batch].detach().movedim(axis + 1, 1)
                roots = condition.axis_masks[batch, axis].movedim(axis, 0)
                size = values.shape[1]
                indices = torch.arange(size - 1, device=probs.device)
                distance = torch.full_like(fixed[:-1], size, dtype=torch.int32)
                for region in regions:
                    patch = (
                        slice(region.row, region.row + region.height),
                        slice(region.col, region.col + region.width),
                    )
                    offset = torch.minimum(
                        (indices - region.index).abs(),
                        (indices + 1 - region.index).abs(),
                    )[:, None, None]
                    distance[:, *patch] = torch.minimum(
                        distance[:, *patch],
                        torch.where(roots[region.index][patch], offset, size),
                    )
                pairs = (distance < size) & ~(fixed[:-1] & fixed[1:])
                positions = torch.stack((indices, indices + 1))
                # Gather before casting so half-precision endpoint gradients are
                # rounded separately, as in the original per-pair computation.
                endpoints = torch.where(
                    fixed[positions, None],
                    observed[:, positions].movedim(0, 2).float().add(1).mul(0.5),
                    values[:, positions].movedim(0, 2).float(),
                ).flatten(3)
                mask = pairs.flatten(1)
                pixels = mask.sum(-1, dtype=torch.float32)
                same = torch.einsum(
                    "pcn,pcn,pn->p", endpoints[0], endpoints[1], mask.float()
                )
                change = 1 - same / pixels.clamp_min(1)
                present = pixels > 0
                if not present.any():
                    continue
                center, tolerance = _reference(change.detach()[present])
                deviation = (change - center).abs()
                penalty = (deviation - tolerance - torch.finfo(change.dtype).eps).relu()
                nearest = (
                    torch.where(pairs, distance, size)
                    .flatten(1)
                    .amin(1)
                    .clamp_max(length - 1)
                    .long()
                )
                totals = [
                    total.index_add(0, nearest, value * pixels)
                    for total, value in zip(
                        totals, (penalty, deviation.detach(), tolerance, 1.0)
                    )
                ]
                reference_axes = reference_axes + 1
    loss_sum, deviation_sum, tolerance_sum, pixels = totals
    deviation = deviation_sum / pixels.clamp_min(1)
    peak = deviation.argmax()
    count = pixels.sum()
    diagnostics = {
        "anchor_neighborhood/deviation": deviation_sum.sum() / count.clamp_min(1),
        "anchor_neighborhood/tolerance": tolerance_sum.sum() / count.clamp_min(1),
        "anchor_neighborhood/pairs": count,
        "anchor_neighborhood/peak_deviation": deviation[peak],
        "anchor_neighborhood/peak_distance": peak.float(),
        "anchor_neighborhood/reference_axes": reference_axes,
    }
    return zero + (loss_sum / pixels.clamp_min(1)).sum(), diagnostics


def _reference(change):
    """Median change rate and the deviation a pair may have from it."""
    center = change.median()
    spread = (change - center).abs().median()
    return center, spread * OUTLIER_SCALE
