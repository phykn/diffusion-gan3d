from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src import AXES
from src.anchor import AnchorCondition


@dataclass(frozen=True)
class AnchorLoss:
    total: torch.Tensor
    coarse: torch.Tensor
    pixel: torch.Tensor
    accuracy: torch.Tensor
    visible_voxels: torch.Tensor


class SoftAnchorLoss(nn.Module):
    def __init__(self, pool_size: int, pixel_weight: float) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.pixel_weight = float(pixel_weight)

    def forward(
        self,
        logits: torch.Tensor,
        condition: AnchorCondition,
        visible: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        observed_axis_masks: torch.Tensor | None = None,
    ) -> AnchorLoss:
        probs = logits.float().softmax(dim=1)
        visibility = visible.to(logits.device).reshape(-1, 1, 1, 1, 1)
        if observed_mask is None:
            observed_mask = condition.mask
            observed_axis_masks = condition.axis_masks
        assert observed_axis_masks is not None
        observed_mask = observed_mask & visibility
        pixel, accuracy, visible_voxels = self.compute_pixel_loss(
            logits,
            (condition.image.float() + 1.0) * 0.5,
            observed_mask,
        )

        target = (condition.image.float() + 1.0) * 0.5
        groups = (
            observed_axis_masks,
            condition.axis_masks & ~observed_axis_masks,
        )
        coarse_losses = [
            self.compute_coarse_loss(
                probs,
                target,
                axis_masks,
                visibility,
            )
            for axis_masks in groups
        ]
        sums, coverage = (torch.stack(values) for values in zip(*coarse_losses))
        valid = coverage > 0
        coarse = (
            (sums / coverage.clamp_min(1e-8)) * valid
        ).sum() / valid.sum().clamp_min(1)
        total = coarse + self.pixel_weight * pixel
        return AnchorLoss(total, coarse, pixel, accuracy, visible_voxels)

    def compute_pixel_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = mask[:, 0]
        visible_voxels = selected.sum()
        denominator = visible_voxels.clamp_min(1)
        pixel = (
            -(target * logits.float().log_softmax(1)).sum(1) * selected
        ).sum() / denominator
        accuracy = (
            (logits.argmax(1) == target.argmax(1)) * selected
        ).sum() / denominator
        return pixel, accuracy, visible_voxels

    def compute_coarse_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        axis_masks: torch.Tensor,
        visibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = probs.sum().mul(0.0)
        coarse_sum = zero
        coarse_coverage = zero
        for axis in AXES:
            axis_mask = axis_masks[:, axis].unsqueeze(1) & visibility
            kernel = [self.pool_size, self.pool_size, self.pool_size]
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
            pooled_target = self.pool_with_mask(
                target,
                axis_mask,
                denominator,
                kernel,
            )
            pooled_probs = self.pool_with_mask(
                probs,
                axis_mask,
                denominator,
                kernel,
            )
            cross_entropy = -(
                pooled_target
                * pooled_probs.clamp_min(torch.finfo(pooled_probs.dtype).eps).log()
            ).sum(dim=1)
            coarse_sum = coarse_sum + (cross_entropy * coverage).sum()
            coarse_coverage = coarse_coverage + coverage.sum()

        return coarse_sum, coarse_coverage

    def pool_with_mask(
        self,
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
