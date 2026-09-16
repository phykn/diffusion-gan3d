import math
from collections.abc import Mapping, Sequence

import torch

from src.plane import PLANE_DIRECTIONS, PLANES, get_axis


class CriticAugment:
    def __init__(
        self,
        mode: bool | str = False,
        prob: float = 1.0,
        planes: Mapping | None = None,
        thickness_axis: str | None = None,
    ) -> None:
        if mode is True:
            raise ValueError("augment true is not supported; use isotropic.")
        self.mode = None if mode is False else mode.strip().lower()
        if self.mode not in (None, "isotropic", "anisotropic"):
            raise ValueError("augment must be false, isotropic, or anisotropic.")
        self.prob = float(prob)
        if not math.isfinite(self.prob) or not 0 <= self.prob <= 1:
            raise ValueError("augmentation probability must be between zero and one.")
        if thickness_axis not in (None, "x", "y", "z"):
            raise ValueError("data.thickness_axis must be x, y, z or null.")
        self.plane_transforms = None
        if planes is not None:
            if mode is not False:
                raise ValueError(
                    "use augmentation.planes instead of augmentation.mode, not both."
                )
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
                if thickness_axis in flips or (
                    rotate and thickness_axis in (rows, cols)
                ):
                    raise ValueError(
                        f"{plane} augmentation must preserve thickness axis {thickness_axis}."
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
        elif thickness_axis is not None and self.mode is not None:
            raise ValueError(
                "declare augmentation.planes when data.thickness_axis is set."
            )
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
        if (
            (self.mode is None and self.plane_transforms is None)
            or self.prob <= 0.0
            or first.shape[0] == 0
        ):
            return tensors

        if self.plane_transforms is None:
            transforms = self.sample_transforms(
                first.shape[0],
                device=first.device,
                square=first.shape[-2] == first.shape[-1],
            )
        else:
            if plane is None:
                raise ValueError("plane is required for plane-specific augmentation.")
            axes = (
                plane.to(device=first.device)
                if isinstance(plane, torch.Tensor)
                else torch.full((first.shape[0],), get_axis(plane), device=first.device)
            )
            if axes.shape != (first.shape[0],) or axes.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(
                    "plane axes must be an integer vector matching the batch."
                )
            transforms = torch.zeros(
                first.shape[0], device=first.device, dtype=torch.long
            )
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
                if first.shape[-2] != first.shape[-1]:
                    allowed = tuple(index for index in allowed if index % 2 == 0)
                mask = axes == axis
                count = first.shape[0]
                choices = torch.tensor(
                    tuple(index for index in allowed if index != 0) or (0,),
                    device=first.device,
                )
                selected = choices[
                    torch.randint(len(choices), (count,), device=first.device)
                ]
                selected.masked_fill_(
                    torch.rand(count, device=first.device) >= self.prob, 0
                )
                transforms = torch.where(mask, selected, transforms)
        return tuple(self.apply_transforms(tensor, transforms) for tensor in tensors)

    def sample_transforms(
        self,
        batch: int,
        device: torch.device,
        square: bool = True,
    ) -> torch.Tensor:
        if self.mode is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        if self.mode == "anisotropic":
            selected = torch.randint(2, (batch,), device=device) * 4
        elif not square:
            selected = torch.randint(4, (batch,), device=device) * 2
        else:
            selected = torch.randint(8, (batch,), device=device)

        selected.masked_fill_(
            torch.rand(batch, device=device) >= self.prob,
            0,
        )
        return selected

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
