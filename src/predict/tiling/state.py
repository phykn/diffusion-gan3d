import math

import torch

from src.phase import validate_num_phases
from src.predict.convert import labels_from_channels, owned_clean_to_probs_


class VolumeState:
    def __init__(
        self,
        num_phases: int,
        shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        self.values = torch.empty(
            (1, num_phases, *shape),
            device=device,
            dtype=torch.float16,
        )

    def read(
        self,
        region: tuple[slice, slice, slice],
    ) -> torch.Tensor:
        return self.values[(slice(None), slice(None), *region)]

    def write(
        self,
        region: tuple[slice, slice, slice],
        values: torch.Tensor,
    ) -> None:
        self.values[(slice(None), slice(None), *region)].copy_(
            values.to(device=self.values.device, dtype=torch.float16)
        )


class TileBuffer:
    def __init__(
        self,
        num_phases: int,
        tile_size: int | tuple[int, int, int],
        enabled: bool,
    ) -> None:
        self.upload: torch.Tensor | None = None
        self.download: torch.Tensor | None = None
        self.workspace: torch.Tensor | None = None
        self.capacity = num_phases * (
            tile_size**3 if isinstance(tile_size, int) else math.prod(tile_size)
        )
        if enabled:
            try:
                self.upload = torch.empty(
                    self.capacity,
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self.download = torch.empty(
                    self.capacity,
                    dtype=torch.float32,
                    pin_memory=True,
                )
            except RuntimeError:
                self.upload = None
                self.download = None

    def read(
        self,
        state: VolumeState,
        region: tuple[slice, slice, slice],
        device: torch.device,
    ) -> torch.Tensor:
        source = state.read(region)
        shape = source.shape
        numel = math.prod(shape)
        if state.values.device != device and self.upload is not None:
            values = self.upload[:numel].view(shape)
        else:
            if (
                self.workspace is None
                or self.workspace.device != state.values.device
                or self.workspace.numel() < numel
            ):
                self.workspace = torch.empty(
                    max(self.capacity, numel),
                    device=state.values.device,
                    dtype=torch.float32,
                )
            values = self.workspace[:numel].view(shape)

        values.copy_(source)
        if values.device == device:
            return values
        return values.to(
            device=device,
            dtype=torch.float32,
            non_blocking=self.upload is not None,
        )

    def stage(self, values: torch.Tensor, device: torch.device) -> torch.Tensor:
        if values.device == device:
            return values
        if (
            device.type == "cpu"
            and self.download is not None
            and values.numel() <= self.download.numel()
        ):
            downloaded = self.download[: values.numel()].view(values.shape)
            downloaded.copy_(values)
            return downloaded
        return values.to(device)


def write_output(
    output: torch.Tensor,
    target: tuple[slice, slice, slice],
    clean: torch.Tensor,
    margin: int = 0,
) -> None:
    validate_num_phases(clean.shape[1])
    source, destination = [], []
    for part, size, length in zip(
        target, output.shape[-3:], clean.shape[-3:], strict=True
    ):
        start = 0 if part.start is None else part.start
        stop = start + length if part.stop is None else part.stop
        left, right = max(start, margin), min(stop, margin + size)
        if left >= right:
            return
        source.append(slice(left - start, right - start))
        destination.append(slice(left - margin, right - margin))
    values = clean[(slice(None), slice(None), *source)]
    if output.ndim == 3:
        output[tuple(destination)].copy_(labels_from_channels(values.squeeze(0)))
    else:
        # Write final probabilities before the FP16 diffusion state can round ties.
        probs = owned_clean_to_probs_(values.clone()).squeeze(0)
        output[(slice(None), *destination)].copy_(probs)
