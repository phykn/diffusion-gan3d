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
    target: torch.Tensor
    planes: int
    conflicts: int
    source_voxels: int

    @property
    def conflict_rate(self) -> float:
        return self.conflicts / self.source_voxels


def build_anchors(
    anchors: Sequence[PlaneAnchor],
    batch_size: int,
    num_phases: int,
    volume_size: int,
    device: torch.device,
    dtype: torch.dtype,
    reconcile: bool = False,
) -> AnchorCondition | None:
    if isinstance(anchors, (str, bytes)) or not isinstance(anchors, Sequence):
        raise TypeError("anchors must be a sequence of PlaneAnchor values.")
    anchors = tuple(anchors)
    if not anchors:
        return None
    if any(not isinstance(anchor, PlaneAnchor) for anchor in anchors):
        raise TypeError("anchors must contain only PlaneAnchor values.")

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
    occupied: set[tuple[int, int]] = set()
    conflicts = 0
    source_voxels = 0
    for anchor in anchors:
        if (
            not isinstance(anchor.axis, int)
            or isinstance(anchor.axis, bool)
            or anchor.axis not in (0, 1, 2)
        ):
            raise ValueError("anchor.axis must be one of 0, 1, or 2.")
        if (
            not isinstance(anchor.index, int)
            or isinstance(anchor.index, bool)
            or not 0 <= anchor.index < volume_size
        ):
            raise ValueError("anchor.index is outside the generated volume.")

        key = (anchor.axis, anchor.index)
        if key in occupied:
            raise ValueError("anchor planes must have unique axis/index positions.")
        occupied.add(key)

        img = anchor.image
        if not isinstance(img, torch.Tensor):
            raise TypeError("anchor.image must be a torch.Tensor.")
        if img.ndim == 2:
            img = img.unsqueeze(0).expand(batch_size, -1, -1)
        elif img.ndim == 3:
            if img.shape[0] != batch_size:
                raise ValueError(f"anchor.image batch size must be {batch_size}.")
        else:
            raise ValueError("anchor.image must have shape [H, W] or [B, H, W].")
        height, width = img.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("anchor.image must not be empty.")
        if height > volume_size or width > volume_size:
            raise ValueError("anchor.image must fit inside the generated plane.")

        position = anchor.position
        if position is None:
            row = (volume_size - height) // 2
            col = (volume_size - width) // 2
        else:
            if (
                not isinstance(position, tuple)
                or len(position) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in position
                )
            ):
                raise ValueError("anchor.position must be a pair of integers.")
            row, col = position
            if (
                row < 0
                or col < 0
                or row + height > volume_size
                or col + width > volume_size
            ):
                raise ValueError("anchor.position places the image outside the plane.")
        img = img.to(device=device, dtype=torch.long)
        if int(img.min()) < 0 or int(img.max()) >= num_phases:
            raise ValueError("anchor.image contains a phase outside num_phases.")

        target_plane = target.select(anchor.axis + 1, anchor.index)
        mask_plane = mask.select(anchor.axis + 2, anchor.index).squeeze(1)
        region = (slice(row, row + height), slice(col, col + width))
        target_patch = target_plane[(slice(None), *region)]
        mask_patch = mask_plane[(slice(None), *region)]
        conflict = mask_patch & (target_patch != img)
        if bool(conflict.any().item()) and not reconcile:
            raise ValueError("anchor planes contain conflicting intersections.")
        conflicts += int(conflict.sum().item())
        source_voxels += batch_size * height * width
        # Independent axis datasets cannot define shared lines, so earlier
        # planes own intersections when training reconciles them.
        target_patch.copy_(torch.where(mask_patch, target_patch, img))
        mask_patch.fill_(True)

    img = torch.full(
        (batch_size, num_phases, volume_size, volume_size, volume_size),
        -1.0,
        device=device,
        dtype=dtype,
    )
    img.scatter_(1, target.unsqueeze(1), 1.0)
    img.mul_(mask)
    return AnchorCondition(
        image=img,
        mask=mask,
        target=target,
        planes=len(anchors),
        conflicts=conflicts,
        source_voxels=source_voxels,
    )
