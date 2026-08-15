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
    coarse_weight: float,
    pixel_weight: float,
) -> AnchorLoss:
    """Keep anchor morphology while allowing exact boundaries to move."""
    _validate(
        logits,
        condition,
        visible,
        pool_size,
        coarse_weight,
        pixel_weight,
    )
    probs = logits.float().softmax(dim=1)
    visibility = visible.reshape(-1, 1, 1, 1, 1)
    union_mask = condition.mask & visibility
    zero = logits.sum() * 0.0
    selected = union_mask[:, 0]
    visible_voxels = int(selected.sum().item())
    if visible_voxels:
        pixel_logits = logits.movedim(1, -1)[selected]
        pixel_target = condition.target[selected]
        pixel = F.cross_entropy(pixel_logits, pixel_target)
        accuracy = (
            (pixel_logits.argmax(dim=1) == pixel_target)
            .to(torch.float32)
            .mean()
        )
    else:
        pixel = zero
        accuracy = zero.detach()

    target = torch.zeros_like(probs)
    target.scatter_(1, condition.target.unsqueeze(1), 1.0)
    coarse_sum = zero
    coarse_cells = 0
    for axis in AXES:
        axis_mask = condition.axis_masks[:, axis].unsqueeze(1) & visibility
        if not bool(axis_mask.any()):
            continue
        kernel = [pool_size, pool_size, pool_size]
        kernel[axis] = 1
        kernel = tuple(min(size, logits.shape[index + 2]) for index, size in enumerate(kernel))
        denominator = F.avg_pool3d(
            axis_mask.to(torch.float32),
            kernel,
            stride=kernel,
            ceil_mode=True,
            count_include_pad=False,
        )
        valid = denominator[:, 0] > 0.0
        if not bool(valid.any()):
            continue
        pooled_target = _masked_pool(target, axis_mask, denominator, kernel)
        pooled_probs = _masked_pool(probs, axis_mask, denominator, kernel)
        cross_entropy = -(
            pooled_target
            * pooled_probs.clamp_min(torch.finfo(pooled_probs.dtype).eps).log()
        ).sum(dim=1)
        coarse_sum = coarse_sum + cross_entropy[valid].sum()
        coarse_cells += int(valid.sum().item())

    coarse = zero if coarse_cells == 0 else coarse_sum / coarse_cells
    total = float(coarse_weight) * coarse + float(pixel_weight) * pixel
    return AnchorLoss(total, coarse, pixel, accuracy, visible_voxels)


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
    coarse_weight: float,
    pixel_weight: float,
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
    if visible.shape != (logits.shape[0],) or visible.dtype != torch.bool:
        raise ValueError("anchor visibility must be a boolean tensor with shape [B].")
    if visible.device != logits.device:
        raise ValueError("anchor visibility and logits must be on the same device.")
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size < 1:
        raise ValueError("anchor pool size must be a positive integer.")
    for name, value in (
        ("anchor coarse weight", coarse_weight),
        ("anchor pixel weight", pixel_weight),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative.")
