import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import AXES
from ..anchor import AnchorCondition


@dataclass(frozen=True)
class AnchorLoss:
    total: torch.Tensor
    coarse: torch.Tensor
    pixel: torch.Tensor
    accuracy: torch.Tensor
    visible_voxels: int


def soft_anchor_loss(
    logits: torch.Tensor,
    condition: AnchorCondition,
    visible: torch.Tensor,
    *,
    pool_size: int,
    pixel_weight: float,
    observed_mask: torch.Tensor | None = None,
    observed_axis_masks: torch.Tensor | None = None,
) -> AnchorLoss:
    """Keep anchor morphology while allowing exact boundaries to move."""
    _validate(
        logits,
        condition,
        visible,
        pool_size,
        pixel_weight,
        observed_mask,
        observed_axis_masks,
    )
    probs = logits.float().softmax(dim=1)
    visibility = visible.reshape(-1, 1, 1, 1, 1)
    if observed_mask is None:
        observed_mask = condition.mask
        observed_axis_masks = condition.axis_masks
    assert observed_axis_masks is not None
    observed_mask = observed_mask & visibility
    zero = logits.sum() * 0.0
    selected = observed_mask[:, 0]
    visible_voxels = int(selected.sum().item())
    if visible_voxels:
        pixel_logits = logits.movedim(1, -1)[selected]
        pixel_target = condition.target[selected]
        pixel = F.cross_entropy(pixel_logits, pixel_target)
        accuracy = (pixel_logits.argmax(dim=1) == pixel_target).to(torch.float32).mean()
    else:
        pixel = zero
        accuracy = zero.detach()

    target = torch.zeros_like(probs)
    target.scatter_(1, condition.target.unsqueeze(1), 1.0)
    groups = []
    for axis_masks in (
        observed_axis_masks,
        condition.axis_masks & ~observed_axis_masks,
    ):
        loss = _coarse_loss(
            probs,
            target,
            axis_masks,
            visibility,
            pool_size,
            zero,
        )
        if loss is not None:
            groups.append(loss)
    coarse = zero if not groups else torch.stack(groups).mean()
    total = coarse + float(pixel_weight) * pixel
    return AnchorLoss(total, coarse, pixel, accuracy, visible_voxels)


def _coarse_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    axis_masks: torch.Tensor,
    visibility: torch.Tensor,
    pool_size: int,
    zero: torch.Tensor,
) -> torch.Tensor | None:
    coarse_sum = zero
    coarse_coverage = zero
    for axis in AXES:
        axis_mask = axis_masks[:, axis].unsqueeze(1) & visibility
        if not bool(axis_mask.any()):
            continue
        kernel = [pool_size, pool_size, pool_size]
        kernel[axis] = 1
        kernel = tuple(
            min(size, probs.shape[index + 2]) for index, size in enumerate(kernel)
        )
        denominator = F.avg_pool3d(
            axis_mask.to(torch.float32),
            kernel,
            stride=kernel,
            ceil_mode=True,
            count_include_pad=False,
        )
        coverage = denominator[:, 0]
        valid = coverage > 0.0
        if not bool(valid.any()):
            continue
        pooled_target = _masked_pool(target, axis_mask, denominator, kernel)
        pooled_probs = _masked_pool(probs, axis_mask, denominator, kernel)
        cross_entropy = -(
            pooled_target
            * pooled_probs.clamp_min(torch.finfo(pooled_probs.dtype).eps).log()
        ).sum(dim=1)
        coarse_sum = coarse_sum + (cross_entropy[valid] * coverage[valid]).sum()
        coarse_coverage = coarse_coverage + coverage[valid].sum()

    if not bool(coarse_coverage > 0.0):
        return None
    return coarse_sum / coarse_coverage


def _masked_pool(
    values: torch.Tensor,
    mask: torch.Tensor,
    denominator: torch.Tensor,
    kernel: tuple[int, int, int],
) -> torch.Tensor:
    numerator = F.avg_pool3d(
        values * mask,
        kernel,
        stride=kernel,
        ceil_mode=True,
        count_include_pad=False,
    )
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def _validate(
    logits: torch.Tensor,
    condition: AnchorCondition,
    visible: torch.Tensor,
    pool_size: int,
    pixel_weight: float,
    observed_mask: torch.Tensor | None,
    observed_axis_masks: torch.Tensor | None,
) -> None:
    if logits.ndim != 5 or not logits.is_floating_point():
        raise ValueError("anchor logits must have shape [B, C, D, H, W].")
    if logits.shape[0] != condition.target.shape[0]:
        raise ValueError("anchor logits and targets must have the same batch size.")
    if logits.shape[2:] != condition.target.shape[1:]:
        raise ValueError("anchor logits and targets must have the same volume shape.")
    if condition.mask.shape != (logits.shape[0], 1, *logits.shape[2:]):
        raise ValueError("anchor mask must match the logits volume.")
    if condition.axis_masks.shape != (logits.shape[0], 3, *logits.shape[2:]):
        raise ValueError("anchor axis masks must match the logits volume.")
    if (observed_mask is None) != (observed_axis_masks is None):
        raise ValueError("observed mask and axis masks must be provided together.")
    if observed_mask is not None and observed_axis_masks is not None:
        if (
            observed_mask.shape != condition.mask.shape
            or observed_mask.dtype != torch.bool
        ):
            raise ValueError(
                "observed anchor mask must be boolean and match the condition."
            )
        if (
            observed_axis_masks.shape != condition.axis_masks.shape
            or observed_axis_masks.dtype != torch.bool
        ):
            raise ValueError(
                "observed anchor axis masks must be boolean and match the condition."
            )
        if (
            observed_mask.device != logits.device
            or observed_axis_masks.device != logits.device
        ):
            raise ValueError("observed anchor masks and logits must share a device.")
        if bool((observed_mask & ~condition.mask).any()):
            raise ValueError(
                "observed anchor mask must be contained in the condition mask."
            )
        if bool((observed_axis_masks & ~condition.axis_masks).any()):
            raise ValueError(
                "observed anchor axis masks must be contained in the condition."
            )
        if not torch.equal(
            observed_axis_masks.any(dim=1, keepdim=True),
            observed_mask,
        ):
            raise ValueError("observed anchor masks must describe the same voxels.")
    if visible.shape != (logits.shape[0],) or visible.dtype != torch.bool:
        raise ValueError("anchor visibility must be a boolean tensor with shape [B].")
    if visible.device != logits.device:
        raise ValueError("anchor visibility and logits must be on the same device.")
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size < 1:
        raise ValueError("anchor pool size must be a positive integer.")
    if (
        not isinstance(pixel_weight, (int, float))
        or isinstance(pixel_weight, bool)
        or not math.isfinite(pixel_weight)
        or pixel_weight < 0.0
    ):
        raise ValueError("anchor pixel weight must be finite and non-negative.")


def pool_size_from_downsampling(downsample_factor: int) -> int:
    """Use the encoder scale nearest the geometric midpoint in resolution."""
    if (
        not isinstance(downsample_factor, int)
        or isinstance(downsample_factor, bool)
        or downsample_factor < 1
    ):
        raise ValueError("downsample factor must be a positive integer.")
    exponent = math.ceil(math.log2(downsample_factor) / 2.0)
    return 2**exponent
