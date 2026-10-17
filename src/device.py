from collections.abc import Sequence

import torch
import torch.nn.functional as F


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be CPU, CUDA or MPS.")
    return device


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def wide_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def adaptive_pool(
    values: torch.Tensor, size: Sequence[int], maximum: bool = False
) -> torch.Tensor:
    size = tuple(size)
    spatial = values.ndim - 4
    if values.device.type == "mps":
        kernel = tuple(n // s for n, s in zip(values.shape[2:], size, strict=True))
        if all(k * s == n for k, s, n in zip(kernel, size, values.shape[2:])):
            pools = (
                (F.max_pool2d, F.max_pool3d)
                if maximum
                else (F.avg_pool2d, F.avg_pool3d)
            )
            return pools[spatial](values, kernel)
        return adaptive_pool(values.cpu(), size, maximum).to(values.device)
    pools = (
        (F.adaptive_max_pool2d, F.adaptive_max_pool3d)
        if maximum
        else (F.adaptive_avg_pool2d, F.adaptive_avg_pool3d)
    )
    return pools[spatial](values, size)
