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


def gpu_amp(mixed_precision: bool, device: torch.device) -> bool:
    # Prediction and the training denoiser; MPS fp16 sampling is faster at equal
    # labels. Training critics keep fp32 on MPS (see src/train/trainer.py).
    return mixed_precision and device.type in {"cuda", "mps"}


def assert_all(valid: torch.Tensor, message: str) -> None:
    # CUDA checks asynchronously to avoid a host sync on every call.
    if valid.device.type == "cuda":
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def wide_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def adaptive_pool(values: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    size = tuple(size)
    spatial = values.ndim - 4
    if values.device.type == "mps":
        kernel = tuple(n // s for n, s in zip(values.shape[2:], size, strict=True))
        if all(k * s == n for k, s, n in zip(kernel, size, values.shape[2:])):
            return (F.avg_pool2d, F.avg_pool3d)[spatial](values, kernel)
        return adaptive_pool(values.cpu(), size).to(values.device)
    return (F.adaptive_avg_pool2d, F.adaptive_avg_pool3d)[spatial](values, size)
