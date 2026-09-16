"""SR allocation budgets, including coarse conversion and HR context/halo."""

import math
from dataclasses import dataclass

import psutil
import torch

from src.predict.memory import cuda_memory_budget, require_memory, workspace_bytes
from src.predict.tile import parse_shape


@dataclass(frozen=True)
class SRMemoryEstimate:
    expanded_shape: tuple[int, int, int]
    coarse_bytes: int
    accumulation_bytes: int
    cpu_tile_bytes: int
    model_tile_bytes: int
    output_bytes: int
    cuda_input_bytes: int

    @property
    def cpu_bytes(self) -> int:
        return (
            self.coarse_bytes
            + self.accumulation_bytes
            + self.cpu_tile_bytes
            + self.output_bytes
        )


def estimate_sr_memory(
    low_shape,
    shape,
    num_phases,
    *,
    tile_size=None,
    margin=0,
    label_input=True,
    input_element_size=8,
    input_on_cuda=False,
) -> SRMemoryEstimate:
    """Conservative additional allocations; caller-owned input is already resident.

    SR keeps the complete HR accumulation on CPU. Its model holds just one
    expanded tile, while interpolation also keeps a coarse-voxel halo on CPU.
    """
    low_shape, shape = parse_shape(low_shape), parse_shape(shape)
    if (
        type(num_phases) is not int
        or num_phases < 1
        or type(margin) is not int
        or margin < 0
    ):
        raise ValueError("num_phases must be positive and margin non-negative.")
    if tile_size is not None and (type(tile_size) is not int or tile_size < 1):
        raise ValueError("tile_size must be positive.")
    tiled = tile_size is not None and any(n > tile_size for n in shape)
    lengths = tuple(min(n, tile_size) for n in shape) if tiled else shape
    expanded = tuple(n + 2 * margin for n in lengths)
    tile_voxels = math.prod(expanded)
    low_voxels, high_voxels = math.prod(low_shape), math.prod(shape)
    # Labels: CPU staging + one int64 index per voxel + fp32 phase channels.
    # Fractions: float32 staging plus validation masks and channel sums.
    validation = (num_phases + 6 * input_element_size + 8) * low_voxels
    coarse = (
        low_voxels * (input_element_size + 8 + 4 * num_phases)
        if label_input
        else 4 * num_phases * low_voxels + (0 if input_on_cuda else validation)
    )
    if tiled:
        scale = shape[0] / low_shape[0]
        if not scale.is_integer() or any(
            high != low * scale for high, low in zip(shape, low_shape)
        ):
            raise ValueError("SR tiles require a uniform integer scale.")
        scale = int(scale)
        halo_low = tuple(math.ceil(n / scale) + 2 for n in expanded)
        halo_high = tuple(n * scale for n in halo_low)
        interpolation = math.prod(halo_low) + math.prod(halo_high)
    else:
        interpolation = high_voxels + tile_voxels
    return SRMemoryEstimate(
        expanded_shape=expanded,
        coarse_bytes=coarse,
        accumulation_bytes=4 * (num_phases + 1) * high_voxels if tiled else 0,
        # Interpolation, a returned tile, weighted tile and blend window.
        cpu_tile_bytes=4 * num_phases * (interpolation + 2 * tile_voxels)
        + 4 * math.prod(lengths),
        model_tile_bytes=4 * (12 * num_phases + 4) * tile_voxels,
        # A full-block prediction is cropped into a separate contiguous output.
        output_bytes=0 if tiled else 4 * num_phases * high_voxels,
        # Fraction validation runs on the original input device, before CPU
        # staging. Account for it even if the model uses a different device.
        cuda_input_bytes=validation if input_on_cuda and not label_input else 0,
    )


def check_sr_memory(estimate: SRMemoryEstimate, generator, input_device=None) -> None:
    working = estimate.model_tile_bytes + workspace_bytes(
        generator, estimate.expanded_shape
    )
    cpu_required = estimate.cpu_bytes
    shared_input_device = False
    if generator.device.type == "cuda":
        # Checking their sum is conservative when input/model use one device;
        # with different GPUs check the input's validation allocation separately.
        if estimate.cuda_input_bytes:
            model_index = generator.device.index
            if model_index is None:
                model_index = torch.cuda.current_device()
            input_index = model_index if input_device is None else input_device.index
            if input_index is None:
                input_index = torch.cuda.current_device()
            shared_input_device = model_index == input_index
        require_memory(
            working + (estimate.cuda_input_bytes if shared_input_device else 0),
            cuda_memory_budget(generator.device),
            "SR CUDA",
        )
    else:
        cpu_required += working
    if (
        estimate.cuda_input_bytes
        and input_device is not None
        and not shared_input_device
    ):
        require_memory(
            estimate.cuda_input_bytes, cuda_memory_budget(input_device), "SR input CUDA"
        )
    require_memory(cpu_required, int(psutil.virtual_memory().available * 0.8), "SR RAM")
