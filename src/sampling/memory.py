import math
from dataclasses import dataclass

import psutil
import torch

from src.sampling.sampler import STATE_DTYPE

STORAGES = ("auto", "cpu", "cuda")
USABLE_FRACTION = 0.8
FLOAT_BYTES = 4
STATE_BYTES = torch.finfo(STATE_DTYPE).bits // 8
# Measured on Apple silicon; tests/sampling/test_memory.py repeats it. The
# activation sizes overstate large MPS tiles about threefold, which is safe.
# Allocator blocks and kernels that do not grow with the volume.
MIN_ACTIVATION_BYTES = 256 * 1024**2
# The Metal allocator holds about 1.05 GiB once a tile exceeds 64^3 voxels.
MPS_HEAP_BYTES = 1280 * 1024**2
# Host memory a GPU runtime takes on its first generation.
HOST_RUNTIME_BYTES = 512 * 1024**2
# fp32 values held per voxel while one tile is processed.
BUFFER_VALUES_PER_PHASE, BUFFER_VALUES = 12, 4
FLUSH_VALUES_PER_PHASE, FLUSH_VALUES = 6, 4
ACTIVATION_VALUES_PER_CHANNEL = 12
GUIDE_VALUES_PER_CHANNEL = 28


@dataclass(frozen=True)
class MemoryEstimate:
    # stored_bytes (state and blend slab) sit on the GPU with storage "cuda" and
    # in RAM otherwise; device_bytes and host_bytes leave them out.
    device_bytes: int
    host_bytes: int
    tiles: int
    shape: tuple
    stored_bytes: int = 0

    def device_total(self, storage):
        return self.device_bytes + (self.stored_bytes if storage == "cuda" else 0)

    def host_total(self, storage):
        return self.host_bytes + (self.stored_bytes if storage != "cuda" else 0)


def estimate_memory(
    plan,
    num_phases,
    channels,
    device,
    *,
    probabilities=False,
    guided=False,
    profile=False,
    coarse=False,
    cfg=False,
    extra_host_bytes=0,
):
    region_voxels = math.prod(plan.region_shape)
    tile_voxels = math.prod(plan.tile_shape)
    output_voxels = math.prod(plan.output_shape)

    state = STATE_BYTES * num_phases * region_voxels
    slab_voxels = plan.tile_shape[0] * math.prod(plan.region_shape[1:])
    slab_values = (2 if guided else 1) * num_phases + 1
    slab = FLOAT_BYTES * slab_values * slab_voxels
    flush = FLOAT_BYTES * (FLUSH_VALUES_PER_PHASE * num_phases + FLUSH_VALUES)
    stored = state + slab + flush * tile_voxels

    buffers = FLOAT_BYTES * (BUFFER_VALUES_PER_PHASE * num_phases + BUFFER_VALUES)
    working = buffers * tile_voxels + _estimate_activations(
        channels, num_phases, tile_voxels, profile or coarse, guided, cfg
    )
    output_bytes = FLOAT_BYTES * num_phases if probabilities else 1
    host = output_bytes * output_voxels + extra_host_bytes
    if device.type == "cpu":
        device_bytes, host_bytes = 0, host + working
    else:
        if device.type == "mps":
            working = max(working, MPS_HEAP_BYTES)
        device_bytes, host_bytes = working, host + HOST_RUNTIME_BYTES
    return MemoryEstimate(
        device_bytes, host_bytes, plan.tile_count, plan.output_shape, stored
    )


def check_memory(estimate, device, storage):
    if storage not in STORAGES:
        raise ValueError(f"storage must be one of {', '.join(STORAGES)}.")
    is_cuda = device.type == "cuda"
    if storage == "cuda" and not is_cuda:
        raise ValueError("storage cuda requires a CUDA device.")
    gpu_budget = get_cuda_budget(device) if is_cuda else 0
    ram_budget = get_ram_budget()

    def count_ram(storage):
        # MPS and the CPU take the device memory from RAM as well.
        shared = 0 if is_cuda else estimate.device_total(storage)
        return estimate.host_total(storage) + shared

    if storage == "auto":
        fits_cuda = (
            is_cuda
            and estimate.device_total("cuda") <= gpu_budget
            and count_ram("cuda") <= ram_budget
        )
        storage = "cuda" if fits_cuda else "cpu"
    if is_cuda:
        _check_fits(estimate.device_total(storage), gpu_budget, "CUDA", estimate)
    _check_fits(count_ram(storage), ram_budget, "RAM", estimate)
    return storage


def get_ram_budget():
    return int(psutil.virtual_memory().available * USABLE_FRACTION)


def get_cuda_budget(device):
    free, _total = torch.cuda.mem_get_info(device)
    # The driver counts cached blocks as used, but this process can reuse them.
    reusable = max(
        0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    return int((free + reusable) * USABLE_FRACTION)


def _estimate_activations(channels, num_phases, voxels, has_volume_input, guided, cfg):
    activations = max(
        MIN_ACTIVATION_BYTES,
        ACTIVATION_VALUES_PER_CHANNEL * FLOAT_BYTES * channels * voxels,
    )
    if has_volume_input:
        activations += FLOAT_BYTES * (num_phases + channels) * voxels
    if guided:
        # The anchor guide keeps the activations for its gradient, per CFG pass.
        passes = 2 if cfg else 1
        activations += (
            passes * GUIDE_VALUES_PER_CHANNEL * FLOAT_BYTES * channels * voxels
        )
    return activations


def _check_fits(required, available, name, estimate):
    if required > available:
        size = "x".join(map(str, estimate.shape))
        raise MemoryError(
            f"estimated {name} memory {required / 1024**3:.2f} GiB exceeds the "
            f"available {available / 1024**3:.2f} GiB for a {size} output in "
            f"{estimate.tiles} tiles"
        )
