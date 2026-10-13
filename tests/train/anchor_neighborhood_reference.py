"""Loop implementation from 0dce898, kept as a numerical regression oracle."""

import torch

from src.anchor import AnchorCondition


def compute_anchor_neighborhood_loss(
    probs: torch.Tensor,
    condition: AnchorCondition,
    radius: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = probs.sum(dtype=torch.float32) * 0.0
    totals = [
        [zero, zero.detach(), zero.detach(), zero.detach()] for _ in range(radius)
    ]
    far_pairs, reference_axes = zero.detach(), zero.detach()
    active = (
        range(len(probs))
        if condition.active_batches is None
        else condition.active_batches
    )
    with torch.autocast(device_type=probs.device.type, enabled=False):
        for batch in active:
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
                length = values.shape[1]
                indices = torch.arange(length - 1, device=probs.device)
                distance = torch.full_like(fixed[:-1], length, dtype=torch.int32)
                candidates = set()
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
                        torch.where(roots[region.index][patch], offset, length),
                    )
                    candidates.update(
                        range(
                            max(0, region.index - radius),
                            min(length - 1, region.index + radius),
                        )
                    )
                eligible = ~(fixed[:-1] & fixed[1:])
                far = (distance >= radius) & (distance < length) & eligible
                target, tolerance, count = _far_reference(values, observed, fixed, far)
                supported = count > 0
                far_pairs = far_pairs + count
                reference_axes = reference_axes + supported
                for index in sorted(candidates):
                    for gap in range(radius):
                        mask = (distance[index] == gap) & eligible[index] & supported
                        joint, pixels = _pair_joint(
                            values, observed, fixed, index, mask
                        )
                        tv = (joint - target).abs().sum() * 0.5
                        penalty = (tv - tolerance - torch.finfo(tv.dtype).eps).relu()
                        loss_sum, tv_sum, tolerance_sum, pairs = totals[gap]
                        totals[gap] = [
                            loss_sum + penalty * pixels,
                            tv_sum + tv.detach() * pixels,
                            tolerance_sum + tolerance * pixels,
                            pairs + pixels,
                        ]
    diagnostics = {
        "anchor_neighborhood/far_pairs": far_pairs,
        "anchor_neighborhood/reference_axes": reference_axes,
    }
    losses, valid = [], []
    for gap, (loss_sum, tv_sum, tolerance_sum, pairs) in enumerate(totals):
        loss = loss_sum / pairs.clamp_min(1)
        losses.append(loss)
        valid.append(pairs > 0)
        prefix = f"anchor_neighborhood/distance{gap}"
        diagnostics.update(
            {
                f"{prefix}/loss": loss.detach(),
                f"{prefix}/tv": tv_sum / pairs.clamp_min(1),
                f"{prefix}/tolerance": tolerance_sum / pairs.clamp_min(1),
                f"{prefix}/pairs": pairs,
            }
        )
    return torch.stack(losses).sum() / torch.stack(valid).sum().clamp_min(
        1
    ), diagnostics


def _pair_joint(values, observed, fixed, index, mask):
    left, right = (
        torch.where(
            fixed[position],
            observed[:, position].float().add(1).mul(0.5),
            values[:, position].float(),
        )
        for position in (index, index + 1)
    )
    count = mask.sum(dtype=torch.float32)
    joint = torch.einsum("chw,dhw,hw->cd", left, right, mask.float())
    return joint / count.clamp_min(1), count


@torch.no_grad()
def _far_reference(values, observed, fixed, mask):
    joints, counts = [], []
    for index in range(len(mask)):
        joint, count = _pair_joint(values, observed, fixed, index, mask[index])
        joints.append(joint)
        counts.append(count)
    if not joints:
        return (
            values.new_zeros((values.shape[0], values.shape[0]), dtype=torch.float32),
            values.new_zeros((), dtype=torch.float32),
            values.new_zeros((), dtype=torch.float32),
        )
    joints, counts = torch.stack(joints), torch.stack(counts)
    count = counts.sum()
    total = (joints * counts[:, None, None]).sum(0)
    target = total / count.clamp_min(1)
    other = count - counts
    leave_one_out = (total - joints * counts[:, None, None]) / other.clamp_min(1)[
        :, None, None
    ]
    deviation = (joints - leave_one_out).abs().sum((1, 2)) * 0.5
    weights = counts * (other > 0)
    tolerance = (deviation * weights).sum() / weights.sum().clamp_min(1)
    return target, tolerance, count
