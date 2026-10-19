import torch

from src.anchor import AnchorCondition
from src.plane import AXIS_NAMES, PLANE_DIRECTIONS, PLANES
from src.train.loss.transition import TransitionStatistics


@torch.no_grad()
def reference_transition(statistics, axis, gap, sign, intervals=None):
    """Conditional transitions and mean crop-to-mean TV, at matching heights."""
    direction = AXIS_NAMES[axis]
    joints, weights = [], []
    for (plane, dim, distance), stats in statistics.items():
        if distance != gap or PLANE_DIRECTIONS[PLANES[plane]][dim - 2] != direction:
            continue
        values = stats.values
        if intervals is None:
            # Pool both orientations only when absolute height is not a condition.
            values = (values + values.transpose(-1, -2)) * 0.5
            joints.append(values.mean(1).unsqueeze(0))
            weights.append(values.new_ones(1, len(values)))
        elif stats.low is not None:
            low, high = intervals
            overlap = (
                torch.minimum(high[:, None, None], stats.high[None])
                - torch.maximum(low[:, None, None], stats.low[None])
            ).clamp_min(0)
            count = overlap.sum(-1)
            if sign < 0:
                values = values.transpose(-1, -2)
            joint = torch.einsum("qkr,krab->qkab", overlap, values)
            joints.append(joint / count.clamp_min(1e-8)[..., None, None])
            weights.append(count)
    if not joints:
        return None
    joint, weight = torch.cat(joints, 1), torch.cat(weights, 1)
    mean = (joint * weight[..., None, None]).sum(1)
    mean = mean / weight.sum(1).clamp_min(1e-8)[:, None, None]
    mass = mean.sum(-1)
    conditional = mean / mass.clamp_min(1e-8)[..., None]
    crop_mass = joint.sum(-1)
    crop_conditional = joint / crop_mass.clamp_min(1e-8)[..., None]
    deviation = (crop_conditional - conditional[:, None]).abs().sum(-1) * 0.5
    crop_weight = weight[..., None] * (crop_mass > 1e-8)
    tolerance = (deviation * crop_weight).sum(1)
    tolerance = tolerance / crop_weight.sum(1).clamp_min(1e-8)
    return conditional, tolerance, mass > 1e-8


def compute_anchor_continuity_loss(
    probs: torch.Tensor,
    measured: AnchorCondition,
    condition: AnchorCondition,
    statistics: dict[tuple[int, int, int], TransitionStatistics],
    max_gap: int,
    height: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare measured-to-unknown joints at every gap without a 3D target.

    Conditions must already reflect classifier-free visibility. Only measured
    regions supply roots; the full condition excludes measured and replay planes
    from destinations. Height matching uses pair-midpoint cell intervals, as in
    the real transition loss. Each crop retains its own conditional distribution
    for the data-derived TV allowance.
    """
    zero = probs.sum(dtype=torch.float32) * 0.0
    near = torch.zeros_like(condition.mask)
    active = (
        tuple(range(len(probs)))
        if measured.active_batches is None
        else measured.active_batches
    )
    diagnostics = {}
    losses, valid_gaps = [], []
    # Reused for both sides of every plane when height is absent.
    references = {}
    with torch.autocast(device_type=probs.device.type, enabled=False):
        for gap in range(1, min(max_gap + 1, max(probs.shape[2:]))):
            loss_sum, tv_sum, tolerance_sum = zero, zero.detach(), zero.detach()
            count = zero.detach()
            sharpness_sum, neighbor_count = zero.detach(), zero.detach()
            for region in measured.regions:
                axis, index = region.axis, region.index
                patch = (
                    slice(region.row, region.row + region.height),
                    slice(region.col, region.col + region.width),
                )
                for batch in active:
                    anchor = (
                        measured.image[batch]
                        .select(axis + 1, index)[..., *patch]
                        .detach()
                        .float()
                        .add(1)
                        .mul(0.5)
                    )
                    observed = measured.mask[batch, 0].select(axis, index)[patch]
                    for sign in (-1, 1):
                        neighbor = index + sign * gap
                        if not 0 <= neighbor < probs.shape[axis + 2]:
                            continue
                        fixed = condition.mask[batch, 0].select(axis, neighbor)[patch]
                        mask = observed & ~fixed
                        values = probs[batch].select(axis + 1, neighbor)[..., *patch]
                        values = values.float()
                        near[batch, 0].select(axis, neighbor)[patch] |= mask
                        sharpness_sum = (
                            sharpness_sum
                            + (values.detach().square().sum(0) * mask).sum()
                        )
                        neighbor_count = neighbor_count + mask.sum()

                        intervals = None
                        source, generated, valid = anchor, values, mask
                        if height is None or axis == 0:
                            source = source.flatten(1).unsqueeze(1)
                            generated = generated.flatten(1).unsqueeze(1)
                            valid = valid.flatten().unsqueeze(0)
                        if height is not None:
                            z = height[batch, 0, :, 0, 0].detach().float()
                            if len(z) < 2:
                                continue
                            width = (z[1:] - z[:-1]).mean()
                            centers = (
                                ((z[index] + z[neighbor]) * 0.5).reshape(1)
                                if axis == 0
                                else z[patch[0]]
                            )
                            intervals = (centers - width * 0.5, centers + width * 0.5)
                            reference = reference_transition(
                                statistics, axis, gap, sign, intervals
                            )
                        else:
                            key = axis, gap
                            if key not in references:
                                references[key] = reference_transition(
                                    statistics, axis, gap, sign
                                )
                            reference = references[key]
                        if reference is None:
                            continue
                        conditional, deviation, supported = reference
                        pixels = valid.sum(-1)
                        joint = (
                            torch.einsum(
                                "arw,brw,rw->rab", source, generated, valid.float()
                            )
                            / pixels.clamp_min(1)[:, None, None]
                        )
                        fraction = (source * valid).sum(-1).T
                        fraction = fraction / pixels.clamp_min(1)[:, None]
                        target = fraction[..., None] * conditional
                        tv = ((joint - target).abs() * supported[..., None]).sum(
                            (-1, -2)
                        ) * 0.5
                        allowance = (fraction * deviation).sum(-1)
                        weight = pixels * ((fraction * supported).sum(-1) > 0)
                        penalty = (tv - allowance).clamp_min(0)
                        loss_sum = loss_sum + (penalty * weight).sum()
                        tv_sum = tv_sum + (tv.detach() * weight).sum()
                        tolerance_sum = tolerance_sum + (allowance * weight).sum()
                        count = count + weight.sum()
            loss = loss_sum / count.clamp_min(1)
            losses.append(loss)
            valid_gaps.append(count > 0)
            prefix = f"anchor_continuity/gap{gap}"
            diagnostics.update(
                {
                    f"{prefix}/loss": loss.detach(),
                    f"{prefix}/tv": tv_sum / count.clamp_min(1),
                    f"{prefix}/tolerance": tolerance_sum / count.clamp_min(1),
                    f"{prefix}/pairs": count,
                    f"{prefix}/sharpness": _measured_mean(
                        sharpness_sum, neighbor_count
                    ),
                }
            )
    diagnostics.update(_sharpness_metrics(probs, near, condition.mask, active))
    if not losses:
        return zero, diagnostics
    return torch.stack(losses).sum() / torch.stack(valid_gaps).sum().clamp_min(
        1
    ), diagnostics


def _measured_mean(total, count):
    return torch.where(
        count > 0, total / count.clamp_min(1), total.new_tensor(float("nan"))
    )


@torch.no_grad()
def _sharpness_metrics(probs, near, fixed, active):
    totals = [probs.new_zeros((), dtype=torch.float32) for _ in range(4)]
    for batch in active:
        # Accumulate plane by plane without another full floating-point volume.
        for index in range(probs.shape[2]):
            sharpness = probs[batch, :, index].float().square().sum(0)
            close = near[batch, 0, index]
            far = ~close & ~fixed[batch, 0, index]
            totals[0] += (sharpness * close).sum()
            totals[1] += close.sum()
            totals[2] += (sharpness * far).sum()
            totals[3] += far.sum()
    near_value = _measured_mean(totals[0], totals[1])
    far_value = _measured_mean(totals[2], totals[3])
    return {
        "anchor_continuity/sharpness_near": near_value,
        "anchor_continuity/sharpness_far": far_value,
        "anchor_continuity/sharpness_difference": near_value - far_value,
        "anchor_continuity/near_voxels": totals[1],
        "anchor_continuity/far_voxels": totals[3],
    }
