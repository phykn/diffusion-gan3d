from collections.abc import Sequence

import torch


class CriticAugment:
    def __init__(
        self,
        mode: bool | str = False,
        prob: float = 1.0,
    ) -> None:
        if mode is True:
            raise ValueError("augment true is not supported; use isotropic.")
        self.mode = None if mode is False else mode.strip().lower()
        if self.mode not in (None, "isotropic", "anisotropic"):
            raise ValueError("augment must be false, isotropic, or anisotropic.")
        self.prob = float(prob)
        self._index_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}

    def apply_together(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        tensors = tuple(inputs)
        first = tensors[0]
        if any(
            tensor.shape[:-3] != first.shape[:-3]
            or tensor.shape[-2:] != first.shape[-2:]
            for tensor in tensors[1:]
        ):
            raise ValueError("augmentation inputs must have matching shapes.")
        if self.mode is None or self.prob <= 0.0 or first.shape[0] == 0:
            return tensors

        transforms = self.sample_transforms(
            first.shape[0],
            device=first.device,
            square=first.shape[-2] == first.shape[-1],
        )
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
        width: int,
    ) -> torch.Tensor:
        key = (device, height, width)
        maps = self._index_cache.get(key)
        if maps is None:
            source = torch.arange(height * width, device=device).reshape(height, width)
            maps = []
            for index in range(8):
                grid = torch.flip(source, dims=(-1,)) if index >= 4 else source
                maps.append(
                    torch.rot90(grid, index % 4, dims=(-2, -1)).reshape(-1)
                )
            maps = torch.stack(maps)
            self._index_cache[key] = maps
        return maps
