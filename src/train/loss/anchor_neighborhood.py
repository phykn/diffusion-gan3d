import torch

from src.anchor import AnchorCondition


def compute_anchor_neighborhood_loss(
    probs: torch.Tensor,
    condition: AnchorCondition,
    radius: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = probs.sum(dtype=torch.float32) * 0.0
    totals = [zero.expand(radius)] + [zero.detach().expand(radius) for _ in range(3)]
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
                if not candidates:
                    continue
                index = torch.tensor(sorted(candidates), device=probs.device)
                gaps = torch.arange(radius, device=probs.device)
                mask = (
                    (distance[index, None] == gaps[None, :, None, None])
                    & eligible[index, None]
                    & supported
                ).flatten(2)
                positions = torch.stack((index, index + 1))
                # Gather before casting so half-precision endpoint gradients are
                # rounded separately, as in the original per-pair computation.
                endpoints = torch.where(
                    fixed[positions, None],
                    observed[:, positions].movedim(0, 2).float().add(1).mul(0.5),
                    values[:, positions].movedim(0, 2).float(),
                ).flatten(3)
                joints, pixels = _pair_joints(endpoints[0], endpoints[1], mask)
                tv = (joints - target).abs().sum((-2, -1)) * 0.5
                penalty = (tv - tolerance - torch.finfo(tv.dtype).eps).relu()
                totals = [
                    totals[0] + (penalty * pixels).sum(0),
                    totals[1] + (tv.detach() * pixels).sum(0),
                    totals[2] + (tolerance * pixels).sum(0),
                    totals[3] + pixels.sum(0),
                ]
    diagnostics = {
        "anchor_neighborhood/far_pairs": far_pairs,
        "anchor_neighborhood/reference_axes": reference_axes,
    }
    losses, valid = [], []
    for gap in range(radius):
        loss_sum, tv_sum, tolerance_sum, pairs = (total[gap] for total in totals)
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


def _pair_joints(left, right, mask):
    # Pairs and distance masks are separate batch dimensions; retain each joint
    # until after TV/ReLU, since a joint pooled across pairs changes the loss.
    counts = mask.sum(-1, dtype=torch.float32)
    weighted = left[:, None] * mask[:, :, None]
    joints = weighted @ right[:, None].transpose(-1, -2)
    return joints / counts.clamp_min(1)[..., None, None], counts


@torch.no_grad()
def _far_reference(values, observed, fixed, mask):
    planes = (
        torch.where(fixed[None], observed.float().add(1).mul(0.5), values.float())
        .movedim(1, 0)
        .flatten(2)
    )
    if not len(mask):
        return (
            planes.new_zeros((planes.shape[1], planes.shape[1])),
            planes.new_zeros(()),
            planes.new_zeros(()),
        )
    joints, counts = _pair_joints(planes[:-1], planes[1:], mask.flatten(1)[:, None])
    joints, counts = joints[:, 0], counts[:, 0]
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
