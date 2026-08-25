from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlaneAnchor:
    image: torch.Tensor
    axis: int
    index: int
    position: tuple[int, int] | None = None


@dataclass(frozen=True)
class AnchorCondition:
    image: torch.Tensor
    mask: torch.Tensor
    axis_masks: torch.Tensor
    target: torch.Tensor
    planes: int
    conflicts: int
    source_voxels: int

    @property
    def conflict_rate(self) -> float:
        return self.conflicts / self.source_voxels


def encode_anchors(
    anchors: Sequence[PlaneAnchor],
    batch_size: int,
    num_phases: int,
    volume_size: int,
    device: torch.device,
    dtype: torch.dtype,
    reconcile: bool = False,
) -> AnchorCondition | None:
    if not anchors:
        return None

    target = torch.zeros(
        batch_size,
        volume_size,
        volume_size,
        volume_size,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        batch_size,
        1,
        volume_size,
        volume_size,
        volume_size,
        dtype=torch.bool,
        device=device,
    )
    axis_masks = torch.zeros(
        batch_size,
        3,
        volume_size,
        volume_size,
        volume_size,
        dtype=torch.bool,
        device=device,
    )
    conflicts = 0
    source_voxels = 0

    for anchor in anchors:
        if anchor.axis not in (0, 1, 2):
            raise ValueError("anchor.axis must be one of 0, 1, or 2.")
        if not 0 <= anchor.index < volume_size:
            raise ValueError("anchor.index is outside the generated volume.")

        image = anchor.image
        if image.ndim == 2:
            image = image.unsqueeze(0).expand(batch_size, -1, -1)
        if image.ndim != 3 or image.shape[0] != batch_size:
            raise ValueError("anchor.image must have shape [H, W] or [B, H, W].")
        if image.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("anchor.image must use an integer dtype.")

        height, width = image.shape[-2:]
        if height > volume_size or width > volume_size:
            raise ValueError("anchor.image must fit inside the generated plane.")

        row, col = anchor.position or (
            (volume_size - height) // 2,
            (volume_size - width) // 2,
        )
        if (
            row < 0
            or col < 0
            or row + height > volume_size
            or col + width > volume_size
        ):
            raise ValueError("anchor.position places the image outside the plane.")

        image = image.to(device=device, dtype=torch.long)
        lower, upper = torch.aminmax(image)
        if int(lower) < 0 or int(upper) >= num_phases:
            raise ValueError("anchor.image contains a phase outside num_phases.")

        target_plane = target.select(anchor.axis + 1, anchor.index)
        mask_plane = mask.select(anchor.axis + 2, anchor.index).squeeze(1)
        axis_plane = axis_masks[:, anchor.axis].select(anchor.axis + 1, anchor.index)
        target_patch = target_plane[:, row : row + height, col : col + width]
        mask_patch = mask_plane[:, row : row + height, col : col + width]
        axis_patch = axis_plane[:, row : row + height, col : col + width]

        conflict = mask_patch & (target_patch != image)
        if bool(conflict.any()) and not reconcile:
            raise ValueError("anchor planes contain conflicting intersections.")
        conflicts += int(conflict.sum())
        source_voxels += image.numel()
        target_patch.copy_(torch.where(mask_patch, target_patch, image))
        mask_patch.fill_(True)
        axis_patch.fill_(True)

    condition_image = torch.full(
        (batch_size, num_phases, volume_size, volume_size, volume_size),
        -1.0,
        dtype=dtype,
        device=device,
    )
    condition_image.scatter_(1, target.unsqueeze(1), 1.0)
    condition_image = condition_image * mask
    return AnchorCondition(
        image=condition_image,
        mask=mask,
        axis_masks=axis_masks,
        target=target,
        planes=len(anchors),
        conflicts=conflicts,
        source_voxels=source_voxels,
    )
