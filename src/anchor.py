from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .data import encode_labels


@dataclass(frozen=True)
class PlaneAnchor:
    labels: torch.Tensor
    axis: int
    index: int


@dataclass(frozen=True)
class AnchorCondition:
    image: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor


def build_anchors(
    anchors: Sequence[PlaneAnchor],
    *,
    batch_size: int,
    num_phases: int,
    volume_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AnchorCondition | None:
    if isinstance(anchors, (str, bytes)) or not isinstance(anchors, Sequence):
        raise TypeError("anchors must be a sequence of PlaneAnchor values.")
    values = tuple(anchors)
    if not values:
        return None
    if any(not isinstance(anchor, PlaneAnchor) for anchor in values):
        raise TypeError("anchors must contain only PlaneAnchor values.")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(num_phases, int) or num_phases < 2:
        raise ValueError("num_phases must be an integer of at least 2.")
    if not isinstance(volume_size, int) or volume_size < 1:
        raise ValueError("volume_size must be a positive integer.")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point.")

    labels = torch.zeros(
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
    occupied_planes: set[tuple[int, int]] = set()
    for anchor in values:
        _check_position(anchor, volume_size)
        key = (anchor.axis, anchor.index)
        if key in occupied_planes:
            raise ValueError("anchor planes must have unique axis/index positions.")
        occupied_planes.add(key)

        plane_labels = _prepare_labels(
            anchor.labels,
            batch_size=batch_size,
            num_phases=num_phases,
            volume_size=volume_size,
            device=device,
        )
        label_view = labels.select(anchor.axis + 1, anchor.index)
        mask_view = mask.select(anchor.axis + 2, anchor.index).squeeze(1)
        conflict = mask_view & (label_view != plane_labels)
        if bool(conflict.any().item()):
            raise ValueError("anchor planes contain conflicting intersections.")
        label_view.copy_(torch.where(mask_view, label_view, plane_labels))
        mask_view.fill_(True)

    image = encode_labels(labels, num_phases).to(device=device, dtype=dtype)
    image = image * mask.to(dtype=dtype)
    return AnchorCondition(image=image, mask=mask, labels=labels)


def _check_position(anchor: PlaneAnchor, volume_size: int) -> None:
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


def _prepare_labels(
    labels: torch.Tensor,
    *,
    batch_size: int,
    num_phases: int,
    volume_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("anchor.labels must be a torch.Tensor.")
    if labels.dtype != torch.long:
        raise ValueError("anchor.labels must have dtype torch.long.")
    if labels.ndim == 2:
        if labels.shape != (volume_size, volume_size):
            raise ValueError("2D anchor.labels must match the generated plane size.")
        labels = labels.unsqueeze(0).expand(batch_size, -1, -1)
    elif labels.ndim == 3:
        expected = (batch_size, volume_size, volume_size)
        if labels.shape != expected:
            raise ValueError(f"batched anchor.labels must have shape {expected}.")
    else:
        raise ValueError("anchor.labels must have shape [H, W] or [B, H, W].")
    labels = labels.to(device=device)
    if labels.numel() == 0 or int(labels.min()) < 0:
        raise ValueError("anchor.labels must not be empty or negative.")
    if int(labels.max()) >= num_phases:
        raise ValueError("anchor.labels contain a phase outside num_phases.")
    return labels
