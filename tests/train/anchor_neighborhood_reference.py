"""Per-pair loop implementation, kept as a numerical regression oracle."""

import torch

from src.anchor import AnchorCondition
from src.train.loss.anchor_neighborhood import OUTLIER_SCALE


def compute_anchor_neighborhood_loss(
    probs: torch.Tensor,
    condition: AnchorCondition,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = probs.sum(dtype=torch.float32) * 0.0
    length = max(probs.shape[2:])
    totals = [
        [zero, zero.detach(), zero.detach(), zero.detach()] for _ in range(length)
    ]
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
                distance = torch.full_like(fixed[:-1], size, dtype=torch.int32)
                for region in regions:
                    patch = (
                        slice(region.row, region.row + region.height),
                        slice(region.col, region.col + region.width),
                    )
                    for index in range(size - 1):
                        offset = min(
                            abs(index - region.index), abs(index + 1 - region.index)
                        )
                        distance[index][patch] = torch.where(
                            roots[region.index][patch],
                            distance[index][patch].clamp_max(offset),
                            distance[index][patch],
                        )
                pairs = (distance < size) & ~(fixed[:-1] & fixed[1:])
                changes = {}
                for index in range(size - 1):
                    if pairs[index].any():
                        changes[index] = _change(
                            values, observed, fixed, index, pairs[index]
                        )
                if not changes:
                    continue
                rates = torch.stack([change.detach() for change in changes.values()])
                center = rates.median()
                tolerance = (rates - center).abs().median() * OUTLIER_SCALE
                for index, change in changes.items():
                    pixels = pairs[index].sum(dtype=torch.float32)
                    deviation = (change - center).abs()
                    penalty = (
                        deviation - tolerance - torch.finfo(change.dtype).eps
                    ).relu()
                    nearest = min(int(distance[index][pairs[index]].min()), length - 1)
                    loss_sum, deviation_sum, tolerance_sum, total = totals[nearest]
                    totals[nearest] = [
                        loss_sum + penalty * pixels,
                        deviation_sum + deviation.detach() * pixels,
                        tolerance_sum + tolerance * pixels,
                        total + pixels,
                    ]
                reference_axes = reference_axes + 1
    loss = zero
    deviations = []
    for loss_sum, deviation_sum, _, pixels in totals:
        loss = loss + loss_sum / pixels.clamp_min(1)
        deviations.append(deviation_sum / pixels.clamp_min(1))
    deviations = torch.stack(deviations)
    peak = deviations.argmax()
    count = sum(total[3] for total in totals)
    diagnostics = {
        "anchor_neighborhood/deviation": sum(total[1] for total in totals)
        / count.clamp_min(1),
        "anchor_neighborhood/tolerance": sum(total[2] for total in totals)
        / count.clamp_min(1),
        "anchor_neighborhood/pairs": count,
        "anchor_neighborhood/peak_deviation": deviations[peak],
        "anchor_neighborhood/peak_distance": peak.float(),
        "anchor_neighborhood/reference_axes": reference_axes,
    }
    return loss, diagnostics


def _change(values, observed, fixed, index, mask):
    left, right = (
        torch.where(
            fixed[position],
            observed[:, position].float().add(1).mul(0.5),
            values[:, position].float(),
        )
        for position in (index, index + 1)
    )
    same = torch.einsum("chw,chw,hw->", left, right, mask.float())
    return 1 - same / mask.sum(dtype=torch.float32)
