import math
from collections.abc import Mapping, Sequence
from itertools import permutations

import torch

from src.plane import PLANE_DIRECTIONS, PLANES, get_axis


def augment_volumes(
    volumes: torch.Tensor, preserve_height: bool = False
) -> torch.Tensor:
    """Sample spatial symmetries independently, keeping phase fractions intact."""
    shape = volumes.shape[-3:]
    orders = [
        order
        for order in permutations(range(3))
        if (not preserve_height or order[0] == 0)
        and tuple(shape[axis] for axis in order) == shape
    ]
    axes = (2, 3) if preserve_height else (1, 2, 3)
    flip_count = 2 ** len(axes)
    choices = torch.randint(len(orders) * flip_count, (len(volumes),)).tolist()
    result = []
    for volume, choice in zip(volumes, choices, strict=True):
        order = orders[choice // flip_count]
        volume = volume.permute(0, *(axis + 1 for axis in order))
        flips = [axis for bit, axis in enumerate(axes) if choice % flip_count & (1 << bit)]
        result.append(volume.flip(flips) if flips else volume)
    return torch.stack(result)


class CriticAugment:
    def __init__(
        self,
        planes: Mapping | None = None,
        prob: float = 1.0,
        preserve_height: bool = False,
    ) -> None:
        self.prob = float(prob)
        if not math.isfinite(self.prob) or not 0 <= self.prob <= 1:
            raise ValueError("augmentation probability must be between zero and one.")
        self.plane_transforms = None
        if planes is not None:
            if not isinstance(planes, Mapping) or not planes:
                raise ValueError("augmentation.planes must be a non-empty mapping.")
            self.plane_transforms = {}
            for plane, policy in planes.items():
                if plane not in PLANES or not isinstance(policy, Mapping):
                    raise ValueError(
                        "augmentation.planes must map xy, xz or yz to policies."
                    )
                if set(policy) - {"flip_axes", "rotate_90"}:
                    raise ValueError(f"unknown augmentation policy for {plane}.")
                rows, cols = PLANE_DIRECTIONS[plane]
                flips = policy.get("flip_axes", [])
                rotate = policy.get("rotate_90", False)
                if (
                    not isinstance(flips, (list, tuple))
                    or any(axis not in (rows, cols) for axis in flips)
                    or len(set(flips)) != len(flips)
                ):
                    raise ValueError(
                        f"{plane}.flip_axes must list distinct in-plane physical axes."
                    )
                if not isinstance(rotate, bool):
                    raise ValueError(f"{plane}.rotate_90 must be true or false.")
                if preserve_height and (
                    "z" in flips or (rotate and "z" in (rows, cols))
                ):
                    raise ValueError(
                        f"{plane} augmentation must preserve height along z."
                    )
                allowed = {0}
                if cols in flips:
                    allowed.add(4)
                if rows in flips:
                    allowed.add(6)
                if len(flips) == 2:
                    allowed.add(2)
                if rotate:
                    allowed = set(range(8)) if flips else set(range(4))
                self.plane_transforms[get_axis(plane)] = tuple(sorted(allowed))
        self._index_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}

    def apply_together(
        self,
        inputs: Sequence[torch.Tensor],
        plane: str | int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        tensors = tuple(inputs)
        first = tensors[0]
        if any(
            tensor.shape[:-3] != first.shape[:-3]
            or tensor.shape[-2:] != first.shape[-2:]
            for tensor in tensors[1:]
        ):
            raise ValueError("augmentation inputs must have matching shapes.")
        if self.plane_transforms is None or self.prob <= 0.0 or first.shape[0] == 0:
            return tensors

        transforms = self.sample_transforms(
            first.shape[0],
            first.device,
            plane,
            square=first.shape[-2] == first.shape[-1],
        )
        return tuple(self.apply_transforms(tensor, transforms) for tensor in tensors)

    def sample_transforms(
        self,
        batch: int,
        device: torch.device,
        plane: str | int | torch.Tensor | None,
        square: bool = True,
    ) -> torch.Tensor:
        if self.plane_transforms is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        if plane is None:
            raise ValueError("plane is required for plane-specific augmentation.")
        axes = (
            plane.to(device=device)
            if isinstance(plane, torch.Tensor)
            else torch.full((batch,), get_axis(plane), device=device)
        )
        if axes.shape != (batch,) or axes.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("plane axes must be an integer vector matching the batch.")
        transforms = torch.zeros(batch, device=device, dtype=torch.long)
        valid = torch.zeros_like(axes, dtype=torch.bool)
        for axis in self.plane_transforms:
            valid |= axes == axis
        if axes.device.type == "cuda":
            torch._assert_async(
                valid.all(), "missing augmentation policy for plane axis"
            )
        elif not bool(valid.all()):
            raise ValueError("missing augmentation policy for plane axis")
        for axis in self.plane_transforms:
            allowed = self.plane_transforms[axis]
            if not square:
                allowed = tuple(index for index in allowed if index % 2 == 0)
            mask = axes == axis
            choices = torch.tensor(
                tuple(index for index in allowed if index != 0) or (0,),
                device=device,
            )
            selected = choices[torch.randint(len(choices), (batch,), device=device)]
            selected.masked_fill_(torch.rand(batch, device=device) >= self.prob, 0)
            transforms = torch.where(mask, selected, transforms)
        return transforms

    def apply_transforms(
        self,
        inputs: torch.Tensor,
        transforms: torch.Tensor,
    ) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        if height != width:
            valid = (transforms.remainder(2) == 0).all()
            if transforms.device.type == "cuda":
                torch._assert_async(
                    valid, "rectangular inputs require shape-preserving transforms."
                )
            elif not bool(valid):
                raise ValueError(
                    "rectangular inputs require shape-preserving transforms."
                )
        maps = self.get_index_maps(inputs.device, height, width)
        indices = maps.index_select(0, transforms.to(torch.long))
        flattened = inputs.reshape(inputs.shape[0], -1, height * width)
        indices = indices.unsqueeze(1).expand(-1, flattened.shape[1], -1)
        return flattened.gather(2, indices).reshape_as(inputs)

    def get_index_maps(
        self,
        device: torch.device,
        height: int,
        width: int,
    ) -> torch.Tensor:
        key = (device, height, width)
        maps = self._index_cache.get(key)
        if maps is None:
            source = torch.arange(height * width, device=device).reshape(height, width)
            maps = []
            for index in range(8):
                grid = torch.flip(source, dims=(-1,)) if index >= 4 else source
                maps.append(torch.rot90(grid, index % 4, dims=(-2, -1)).reshape(-1))
            maps = torch.stack(maps)
            self._index_cache[key] = maps
        return maps


def crop_images(
    images: torch.Tensor,
    size: int | tuple[int, int],
    centers: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    crop_h, crop_w = (size, size) if isinstance(size, int) else size
    if crop_h < 1 or crop_w < 1:
        raise ValueError("crop size must be a positive integer.")
    height, width = images.shape[-2:]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size must fit inside the images.")
    if (height, width) == (crop_h, crop_w):
        return images

    top = torch.randint(height - crop_h + 1, (images.shape[0],)).tolist()
    left = torch.randint(width - crop_w + 1, (images.shape[0],)).tolist()
    if centers is not None:
        for index, (row, col) in enumerate(centers):
            top[index] = min(max(row - crop_h // 2, 0), height - crop_h)
            left[index] = min(max(col - crop_w // 2, 0), width - crop_w)
    return torch.stack(
        [
            image[..., row : row + crop_h, col : col + crop_w]
            for image, row, col in zip(images, top, left, strict=True)
        ]
    )
