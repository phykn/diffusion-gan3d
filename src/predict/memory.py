"""Allocation estimates for generation; model workspaces remain approximate."""

import math
from dataclasses import dataclass

import psutil
import torch

from src.predict.tile import parse_shape


@dataclass(frozen=True)
class MemoryEstimate:
    shape: tuple[int, int, int]
    generation_shape: tuple[int, int, int]
    tile_size: int
    tile_count: int
    state_bytes: int
    fusion_bytes: int
    buffer_bytes: int
    output_bytes: int

    @property
    def storage_bytes(self) -> int:
        return self.state_bytes + self.fusion_bytes

    @property
    def total_bytes(self) -> int:
        """Conservative tensor budget, excluding network activations/workspaces."""
        return self.storage_bytes + self.buffer_bytes + self.output_bytes


def estimate_memory(
    shape,
    num_phases: int,
    *,
    tile_size: int | None = None,
    margin: int = 0,
    overlap: int = 0,
    probabilities: bool = False,
) -> MemoryEstimate:
    """Count two fp16 states, an fp32 slab, tile buffers and CPU output.

    The result is a budget, not a guarantee: allocator fragmentation and
    convolution workspaces also depend on the loaded network and device.
    """
    shape = parse_shape(shape)
    if not isinstance(num_phases, int) or num_phases < 1:
        raise ValueError("num_phases must be a positive integer.")
    if not isinstance(margin, int) or margin < 0:
        raise ValueError("margin must be a non-negative integer.")
    generation_shape = tuple(size + 2 * margin for size in shape)
    tile_size = max(generation_shape) if tile_size is None else tile_size
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer.")
    if not isinstance(overlap, int) or overlap < 0 or 2 * overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= 2 * overlap < tile_size.")
    stride = tile_size - 2 * overlap
    tile_shape = tuple(min(size, tile_size) for size in generation_shape)
    voxels = math.prod(generation_shape)
    return MemoryEstimate(
        shape=shape,
        generation_shape=generation_shape,
        tile_size=tile_size,
        tile_count=math.prod(
            1 + (max(0, size - tile_size) + stride - 1) // stride
            for size in generation_shape
        ),
        state_bytes=4 * num_phases * voxels,
        fusion_bytes=4
        * (num_phases + 1)
        * min(tile_size, generation_shape[0])
        * math.prod(generation_shape[1:]),
        # Upload/download, prediction, weighting and posterior scratch tensors.
        buffer_bytes=4 * (12 * num_phases + 4) * math.prod(tile_shape),
        output_bytes=(
            4 * num_phases * math.prod(shape)
            if probabilities
            else voxels + math.prod(shape)
        ),
    )


def workspace_bytes(generator, tile_size: int | tuple[int, int, int]) -> int:
    """Reserve activations and convolution scratch separately from storage."""
    channels = getattr(
        getattr(generator.model, "input", None), "out_channels", generator.num_phases
    )
    # FP32 normalization, skip activations and convolution scratch coexist,
    # including when convolutions themselves run under autocast.
    voxels = tile_size**3 if isinstance(tile_size, int) else math.prod(tile_size)
    return max(256 * 1024**2, 12 * 4 * channels * voxels)


def cuda_memory_budget(device) -> int:
    free, _total = torch.cuda.mem_get_info(device)
    reusable = max(
        0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    return int((free + reusable) * 0.8)


def require_memory(required: int, available: int, name: str) -> None:
    if required > available:
        raise MemoryError(
            f"estimated {name} allocation {required / 1024**3:.2f} GiB "
            f"exceeds the available budget {available / 1024**3:.2f} GiB"
        )


def select_storage(storage: str, estimate: MemoryEstimate, generator) -> str:
    if storage not in {"auto", "cuda", "cpu"}:
        raise ValueError("storage must be 'auto', 'cuda', or 'cpu'.")
    device = generator.device
    if storage == "cuda" and device.type != "cuda":
        raise ValueError("storage='cuda' requires a CUDA generator.")
    reserve = workspace_bytes(generator, estimate.tile_size)
    gpu_budget = 0
    if device.type == "cuda":
        # The driver reports allocator-cached blocks as used even though this
        # process can reuse them for the next request without another malloc.
        # Keep headroom for the allocator and concurrent allocations.
        gpu_budget = cuda_memory_budget(device)
    working = estimate.buffer_bytes + reserve
    if storage == "auto":
        storage = (
            "cuda"
            if device.type == "cuda" and estimate.storage_bytes + working <= gpu_budget
            else "cpu"
        )
    gpu_required = working + (estimate.storage_bytes if storage == "cuda" else 0)
    if device.type == "cuda":
        require_memory(gpu_required, gpu_budget, "CUDA")
    cpu_required = estimate.output_bytes + estimate.buffer_bytes
    if storage == "cpu":
        cpu_required += estimate.storage_bytes
    if device.type == "cpu":
        cpu_required += reserve
    cpu_budget = int(psutil.virtual_memory().available * 0.8)
    require_memory(cpu_required, cpu_budget, "RAM")
    return storage
