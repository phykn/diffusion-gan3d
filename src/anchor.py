from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.prepare.resize import phase_channels


@dataclass(frozen=True)
class PlaneAnchor:
    image: torch.Tensor
    axis: int
    index: int
    position: tuple[int, int] | None = None


@dataclass(frozen=True)
class AnchorRegion:
    axis: int
    index: int
    row: int
    col: int
    height: int
    width: int


@dataclass(frozen=True)
class AnchorCondition:
    image: torch.Tensor
    mask: torch.Tensor
    axis_masks: torch.Tensor
    target: torch.Tensor
    planes: int
    conflicts: int
    source_voxels: int
    regions: tuple[AnchorRegion, ...] = ()
    active_batches: tuple[int, ...] | None = None

    @property
    def conflict_rate(self) -> float:
        return self.conflicts / self.source_voxels


def encode_anchors(
    anchors: Sequence[PlaneAnchor],
    batch_size: int,
    num_phases: int,
    volume_size: int | tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    reconcile: bool = False,
    validate: bool = True,
) -> AnchorCondition | None:
    if not anchors:
        return None
    shape = (volume_size,) * 3 if isinstance(volume_size, int) else volume_size
    if (
        len(shape) != 3
        or any(type(size) is not int or size < 1 for size in shape)
        or type(batch_size) is not int
        or batch_size < 1
        or type(num_phases) is not int
        or num_phases < 1
    ):
        raise ValueError(
            "anchor volume shape, batch size and phase count must be positive integers."
        )

    target = torch.zeros(
        batch_size,
        *shape,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        batch_size,
        1,
        *shape,
        dtype=torch.bool,
        device=device,
    )
    axis_masks = torch.zeros(
        batch_size,
        3,
        *shape,
        dtype=torch.bool,
        device=device,
    )
    conflicts = 0
    source_voxels = 0
    regions = []
    condition_image = torch.zeros(
        (batch_size, num_phases, *shape), dtype=dtype, device=device
    )

    for anchor in anchors:
        if type(anchor.axis) is not int or anchor.axis not in (0, 1, 2):
            raise ValueError("anchor.axis must be one of 0, 1, or 2.")
        if type(anchor.index) is not int or not 0 <= anchor.index < shape[anchor.axis]:
            raise ValueError("anchor.index is outside the generated volume.")

        probs = anchor_probabilities(
            anchor.image, batch_size, num_phases, device, dtype, validate
        )
        height, width = probs.shape[-2:]
        plane_shape = tuple(
            size for axis, size in enumerate(shape) if axis != anchor.axis
        )
        if height > plane_shape[0] or width > plane_shape[1]:
            raise ValueError("anchor.image must fit inside the generated plane.")

        if anchor.position is not None and (
            len(anchor.position) != 2
            or any(type(value) is not int for value in anchor.position)
        ):
            raise ValueError("anchor.position must contain two integer coordinates.")
        row, col = (
            anchor.position
            if anchor.position is not None
            else (
                (plane_shape[0] - height) // 2,
                (plane_shape[1] - width) // 2,
            )
        )
        if (
            row < 0
            or col < 0
            or row + height > plane_shape[0]
            or col + width > plane_shape[1]
        ):
            raise ValueError("anchor.position places the image outside the plane.")

        image = probs.argmax(1)

        target_plane = target.select(anchor.axis + 1, anchor.index)
        mask_plane = mask.select(anchor.axis + 2, anchor.index).squeeze(1)
        axis_plane = axis_masks[:, anchor.axis].select(anchor.axis + 1, anchor.index)
        target_patch = target_plane[:, row : row + height, col : col + width]
        mask_patch = mask_plane[:, row : row + height, col : col + width]
        axis_patch = axis_plane[:, row : row + height, col : col + width]
        value_patch = condition_image.select(anchor.axis + 2, anchor.index)[
            :, :, row : row + height, col : col + width
        ]
        encoded = probs.mul(2).sub(1)

        conflict = mask_patch & ((value_patch - encoded).abs().amax(1) > 1e-5)
        conflict_count = int(conflict.sum()) if validate else 0
        if conflict_count and not reconcile:
            raise ValueError("anchor planes contain conflicting intersections.")
        conflicts += conflict_count
        source_voxels += image.numel()
        target_patch.copy_(torch.where(mask_patch, target_patch, image))
        value_patch.copy_(torch.where(mask_patch.unsqueeze(1), value_patch, encoded))
        mask_patch.fill_(True)
        axis_patch.fill_(True)
        regions.append(AnchorRegion(anchor.axis, anchor.index, row, col, height, width))

    return AnchorCondition(
        image=condition_image,
        mask=mask,
        axis_masks=axis_masks,
        target=target,
        planes=len(anchors),
        conflicts=conflicts,
        source_voxels=source_voxels,
        regions=tuple(regions),
    )


def anchor_probabilities(image, batch_size, num_phases, device, dtype, validate):
    if image.numel() == 0:
        raise ValueError("anchor.image must not be empty.")
    if image.dtype.is_floating_point:
        if image.ndim == 3:
            image = image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if image.ndim != 4 or image.shape[:2] != (batch_size, num_phases):
            raise ValueError(
                "anchor phase fractions must have shape [C,H,W] or [B,C,H,W]."
            )
        if validate:
            valid = (
                torch.isfinite(image).all() & (image >= 0).all() & (image <= 1).all()
            )
            valid = valid & ((image.sum(1) - 1).abs() < 1e-5).all()
            if not bool(valid):
                raise ValueError(
                    "anchor phase fractions must be finite, non-negative and sum to one."
                )
        probs = image.to(device=device, dtype=dtype)
    else:
        if image.ndim == 2:
            image = image.unsqueeze(0).expand(batch_size, -1, -1)
        if image.ndim != 3 or image.shape[0] != batch_size:
            raise ValueError("anchor.image must have shape [H, W] or [B, H, W].")
        probs = phase_channels(image, num_phases).to(device=device, dtype=dtype)
    return probs
