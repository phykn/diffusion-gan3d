import math
from dataclasses import dataclass

import psutil
import torch

from src.sampling.plan import TilePlan
from src.sampling.sampler import STATE_DTYPE

# The fraction of free memory a generation may use.
BUDGET = 0.8
# The activation constants below overstate large MPS tiles about threefold,
# which is safe. MPS_HEAP, HOST_RUNTIME and the flush scratch were measured on
# Apple silicon; tests/sampling/test_memory.py repeats that measurement.
# Least activation reserve: allocator blocks and kernels outside the volume.
MIN_RESERVE = 256 * 1024**2
STATE_BYTES = torch.finfo(STATE_DTYPE).bits // 8
# The Metal allocator holds about 1.05 GiB once a tile exceeds 64^3 voxels,
# whatever the tile size up to 144^3.
MPS_HEAP = 1280 * 1024**2
# Host memory a GPU runtime takes on its first generation (kernel libraries,
# caches): about 0.3-0.5 GiB on MPS.
HOST_RUNTIME = 512 * 1024**2
STORAGES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class MemoryEstimate:
    """Peak bytes of one generation.

    device_bytes and host_bytes leave out stored_bytes (state, blend slab and
    the scratch of flushing it), which sit on the GPU with storage cuda and in
    RAM otherwise.
    """

    device_bytes: int
    host_bytes: int
    tiles: int
    shape: tuple[int, int, int]
    stored_bytes: int = 0

    def device_total(self, storage: str) -> int:
        return self.device_bytes + (self.stored_bytes if storage == "cuda" else 0)

    def host_total(self, storage: str) -> int:
        return self.host_bytes + (self.stored_bytes if storage != "cuda" else 0)


def tile_buffer_bytes(phases: int, voxels: int) -> int:
    # Upload/download, prediction, weighting and posterior scratch tensors.
    return 4 * (12 * phases + 4) * voxels


def flush_bytes(phases: int, voxels: int) -> int:
    # Staged predictions, clean values, posterior mean and noise of one tile
    # target, where the state is stored.
    return 4 * (6 * phases + 4) * voxels


def activation_bytes(
    channels: int,
    phases: int,
    voxels: int,
    *,
    profile: bool = False,
    coarse: bool = False,
    guided: bool = False,
    cfg: bool = False,
) -> int:
    """Denoiser activations for one tile of voxels.

    FP32 normalization, skip activations and convolution scratch coexist, also
    when convolutions run under autocast. An anchor guide keeps the activations
    for its input gradient: about 28 fp32 values per first-level channel and
    voxel, for each of the two CFG passes when CFG is on.
    """
    reserve = max(MIN_RESERVE, 12 * 4 * channels * voxels)
    if profile or coarse:
        reserve += 4 * (phases + channels) * voxels
    if guided:
        reserve += (2 if cfg else 1) * 28 * 4 * channels * voxels
    return reserve


def estimate_memory(
    plan: TilePlan,
    phases: int,
    channels: int,
    device: torch.device,
    *,
    probabilities: bool = False,
    guided: bool = False,
    profile: bool = False,
    coarse: bool = False,
    cfg: bool = False,
    extra_host_bytes: int = 0,
) -> MemoryEstimate:
    voxels = math.prod(plan.shape)
    tile_voxels = math.prod(plan.tile_edge)
    output_voxels = math.prod(plan.output_shape)
    state = STATE_BYTES * phases * voxels
    slab = (
        4
        * ((2 if guided else 1) * phases + 1)
        * plan.tile_edge[0]
        * math.prod(plan.shape[1:])
    )
    stored = state + slab + flush_bytes(phases, tile_voxels)
    output = 4 * phases * output_voxels if probabilities else output_voxels
    working = tile_buffer_bytes(phases, tile_voxels) + activation_bytes(
        channels,
        phases,
        tile_voxels,
        profile=profile,
        coarse=coarse,
        guided=guided,
        cfg=cfg,
    )
    host = output + extra_host_bytes
    if device.type == "cpu":
        return MemoryEstimate(
            0, host + working, plan.tile_count, plan.output_shape, stored
        )
    if device.type == "mps":
        working = max(working, MPS_HEAP)
    return MemoryEstimate(
        working, host + HOST_RUNTIME, plan.tile_count, plan.output_shape, stored
    )


def ram_budget() -> int:
    return int(psutil.virtual_memory().available * BUDGET)


def cuda_budget(device: torch.device) -> int:
    free, _total = torch.cuda.mem_get_info(device)
    # The driver counts cached allocator blocks as used, but this process can
    # reuse them without another allocation.
    reusable = max(
        0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    return int((free + reusable) * BUDGET)


def _require(required: int, available: int, name: str, estimate) -> None:
    if required > available:
        size = "x".join(map(str, estimate.shape))
        raise MemoryError(
            f"estimated {name} memory {required / 1024**3:.2f} GiB exceeds the "
            f"available {available / 1024**3:.2f} GiB for a {size} output in "
            f"{estimate.tiles} tiles"
        )


def check_memory(estimate: MemoryEstimate, device: torch.device, storage: str) -> str:
    """The storage to use (auto: cuda when it fits there), or MemoryError.

    On MPS and the CPU the device shares RAM with the host.
    """
    if storage not in STORAGES:
        raise ValueError(f"storage must be one of {', '.join(STORAGES)}.")
    cuda = device.type == "cuda"
    if storage == "cuda" and not cuda:
        raise ValueError("storage cuda requires a CUDA device.")
    gpu = cuda_budget(device) if cuda else 0
    ram = ram_budget()

    def host(choice):
        return estimate.host_total(choice) + (
            0 if cuda else estimate.device_total(choice)
        )

    if storage == "auto":
        storage = (
            "cuda"
            if cuda and estimate.device_total("cuda") <= gpu and host("cuda") <= ram
            else "cpu"
        )
    if cuda:
        _require(estimate.device_total(storage), gpu, "CUDA", estimate)
    _require(host(storage), ram, "RAM", estimate)
    return storage
