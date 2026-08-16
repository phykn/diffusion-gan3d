import ctypes
import math
import os

import torch


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
        key = (slice(None), slice(None), *region)
        return self.values[key]

    def write(
        self,
        region: tuple[slice, slice, slice],
        values: torch.Tensor,
    ) -> None:
        key = (slice(None), slice(None), *region)
        self.values[key].copy_(
            values.to(device=self.values.device, dtype=torch.float16)
        )


class TileBuffer:
    def __init__(
        self,
        num_phases: int,
        tile_size: int,
        enabled: bool,
    ) -> None:
        self.upload: torch.Tensor | None = None
        self.download: torch.Tensor | None = None
        self.workspace: torch.Tensor | None = None
        self.capacity = num_phases * tile_size**3
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


def get_cuda_available_memory(device: torch.device) -> int:
    free, total = torch.cuda.mem_get_info(device)
    reclaimable = max(
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
        0,
    )
    return min(total, free + reclaimable)


def check_cpu_memory(required: int) -> None:
    available = get_available_memory()
    if available is None:
        raise RuntimeError("available CPU memory could not be determined.")
    safe = int(available * 0.8)
    if required > safe:
        raise MemoryError(
            f"planned CPU allocation requires {required} bytes, "
            f"which exceeds 80% of the {available} available bytes."
        )


def get_available_memory() -> int | None:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64),
                ("available_physical", ctypes.c_uint64),
                ("total_page_file", ctypes.c_uint64),
                ("available_page_file", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("available_virtual", ctypes.c_uint64),
                ("available_extended_virtual", ctypes.c_uint64),
            )

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(status)
            )
        except (AttributeError, OSError):
            return None
        return int(status.available_physical) if success else None

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    return pages * page_size
