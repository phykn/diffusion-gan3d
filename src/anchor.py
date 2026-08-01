from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlaneAnchor:
    image: torch.Tensor
    axis: int
    index: int


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
    *,
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
            if img.shape != (volume_size, volume_size):
                raise ValueError("anchor.image must match the generated plane size.")
            img = img.unsqueeze(0).expand(batch_size, -1, -1)
        elif img.ndim == 3:
            shape = (batch_size, volume_size, volume_size)
            if img.shape != shape:
                raise ValueError(f"anchor.image must have shape {shape}.")
        else:
            raise ValueError("anchor.image must have shape [H, W] or [B, H, W].")
        img = img.to(device=device, dtype=torch.long)
        if int(img.min()) < 0 or int(img.max()) >= num_phases:
            raise ValueError("anchor.image contains a phase outside num_phases.")

        target_plane = target.select(anchor.axis + 1, anchor.index)
        mask_plane = mask.select(anchor.axis + 2, anchor.index).squeeze(1)
        conflict = mask_plane & (target_plane != img)
        if bool(conflict.any().item()) and not reconcile:
            raise ValueError("anchor planes contain conflicting intersections.")
        conflicts += int(conflict.sum().item())
        # Independent axis datasets cannot define shared lines, so earlier
        # planes own intersections when training reconciles them.
        target_plane.copy_(torch.where(mask_plane, target_plane, img))
        mask_plane.fill_(True)

    img = (
        F.one_hot(target, num_classes=num_phases)
        .movedim(-1, 1)
        .to(device=device, dtype=dtype)
        .mul_(2.0)
        .sub_(1.0)
    )
    img = img * mask.to(dtype=dtype)
    return AnchorCondition(
        image=img,
        mask=mask,
        target=target,
        planes=len(anchors),
        conflicts=conflicts,
        source_voxels=len(anchors) * batch_size * volume_size * volume_size,
    )
