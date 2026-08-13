import math
from collections.abc import Sequence
from typing import Literal, cast

import torch

from .. import AXES

AugmentMode = Literal["isotropic", "anisotropic"]

MODES = {"isotropic", "anisotropic"}
ALL_TRANSFORMS = tuple(range(8))
SHAPE_PRESERVING_TRANSFORMS = (0, 2, 4, 6)
ANISOTROPIC_TRANSFORMS = (0, 4)


class CriticAugment:
    """Apply differentiable, axis-aware symmetries to critic inputs."""

    def __init__(
        self,
        mode: bool | str = False,
        prob: float = 1.0,
    ) -> None:
        self.mode = self.parse_mode(mode)
        if (
            not isinstance(prob, (int, float))
            or isinstance(prob, bool)
            or not math.isfinite(prob)
            or not 0.0 <= prob <= 1.0
        ):
            raise ValueError("augment_prob must be between zero and one.")
        self.prob = float(prob)
        self._index_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}

    @staticmethod
    def parse_mode(mode: bool | str) -> AugmentMode | None:
        if isinstance(mode, bool):
            if mode:
                raise ValueError("augment true is not supported; use isotropic.")
            return None
        if not isinstance(mode, str):
            raise TypeError("augment must be a boolean or preset name.")
        normalized = mode.strip().lower()
        if normalized not in MODES:
            choices = ", ".join(sorted(MODES))
            raise ValueError(f"augment must be false or one of: {choices}.")
        return cast(AugmentMode, normalized)

    @property
    def enabled(self) -> bool:
        return self.mode is not None and self.prob > 0.0

    def allowed_transforms(self, axis: int) -> tuple[int, ...]:
        self.check_axis(axis)
        if self.mode is None:
            return (0,)
        if self.mode == "isotropic":
            return ALL_TRANSFORMS
        return ANISOTROPIC_TRANSFORMS

    def apply_pair(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        axis: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous, current = self.apply_together((previous, current), axis)
        return previous, current

    def apply_together(
        self,
        inputs: Sequence[torch.Tensor],
        axis: int | torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        tensors = tuple(inputs)
        if not tensors:
            raise ValueError("augmentation inputs must not be empty.")
        first = tensors[0]
        self.check_inputs(first)
        for tensor in tensors[1:]:
            self.check_inputs(tensor)
            if tensor.shape[:-3] != first.shape[:-3]:
                raise ValueError("augmentation inputs must have matching batch axes.")
            if tensor.shape[-2:] != first.shape[-2:]:
                raise ValueError(
                    "augmentation inputs must have matching spatial sizes."
                )
            if tensor.device != first.device or tensor.dtype != first.dtype:
                raise ValueError("augmentation inputs must share device and dtype.")
        if not self.enabled or first.shape[0] == 0:
            return tensors

        axes = self.prepare_axes(axis, first.shape[0], first.device)
        transforms = self.sample_transforms(
            axes,
            square=first.shape[-2] == first.shape[-1],
        )
        return tuple(self.apply_transforms(tensor, transforms) for tensor in tensors)

    def sample_transforms(
        self, axes: torch.Tensor, square: bool = True
    ) -> torch.Tensor:
        batch = axes.shape[0]
        device = axes.device
        if self.mode == "anisotropic":
            selected = torch.randint(2, (batch,), device=device).mul_(4)
        elif not square:
            selected = torch.randint(4, (batch,), device=device).mul_(2)
        else:
            selected = torch.randint(8, (batch,), device=device)

        active = torch.rand(batch, device=device) < self.prob
        return torch.where(active, selected, torch.zeros_like(selected))

    def apply_transforms(
        self,
        inputs: torch.Tensor,
        transforms: torch.Tensor,
    ) -> torch.Tensor:
        if transforms.shape != (inputs.shape[0],):
            raise ValueError("transforms must have one value per batch item.")
        if transforms.device != inputs.device:
            raise ValueError("transforms must be on the input device.")
        height, width = inputs.shape[-2:]
        if height != width and bool((transforms.remainder(2) != 0).any()):
            raise ValueError("rectangular inputs require shape-preserving transforms.")
        maps = self.get_index_maps(inputs.device, height, width)
        indices = maps.index_select(0, transforms.to(torch.long))
        flattened = inputs.reshape(inputs.shape[0], -1, height * width)
        indices = indices.unsqueeze(1).expand(-1, flattened.shape[1], -1)
        return flattened.gather(2, indices).reshape_as(inputs)

    def get_index_maps(
        self,
        device: torch.device,
        height: int,
        width: int | None = None,
    ) -> torch.Tensor:
        width = height if width is None else width
        key = (device, height, width)
        maps = self._index_cache.get(key)
        if maps is None:
            source = torch.arange(height * width, device=device).reshape(height, width)
            maps = torch.stack(
                [self.transform(source, index).reshape(-1) for index in range(8)]
            )
            self._index_cache[key] = maps
        return maps

    @staticmethod
    def transform(inputs: torch.Tensor, index: int) -> torch.Tensor:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 8:
            raise ValueError("transform index must be between zero and seven.")
        if index >= 4:
            inputs = torch.flip(inputs, dims=(-1,))
        return torch.rot90(inputs, index % 4, dims=(-2, -1))

    @staticmethod
    def prepare_axes(
        axis: int | torch.Tensor,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(axis, int) and not isinstance(axis, bool):
            CriticAugment.check_axis(axis)
            return torch.full((batch,), axis, device=device, dtype=torch.long)
        if not isinstance(axis, torch.Tensor):
            raise TypeError("axis must be an integer or tensor.")
        if axis.shape != (batch,):
            raise ValueError("axis tensor must have one value per batch item.")
        axes = axis.to(device=device, dtype=torch.long)
        if axes.numel() and (int(axes.min()) < 0 or int(axes.max()) > 2):
            raise ValueError("axis values must be zero, one, or two.")
        return axes

    @staticmethod
    def check_axis(axis: int) -> None:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis not in AXES:
            raise ValueError("axis must be zero, one, or two.")

    @staticmethod
    def check_inputs(inputs: torch.Tensor) -> None:
        if inputs.ndim < 4:
            raise ValueError(
                "augmentation inputs need batch, channel, and spatial dimensions."
            )
