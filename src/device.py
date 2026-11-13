import torch
import torch.nn.functional as F


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_device(device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be CPU, CUDA or MPS.")
    return device


def gpu_amp(mixed_precision, device):
    return mixed_precision and device.type in {"cuda", "mps"}


def assert_all(valid, message):
    # CUDA checks asynchronously to avoid a host sync on every call.
    if valid.device.type == "cuda":
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def wide_dtype(device):
    return torch.float32 if device.type == "mps" else torch.float64


def adaptive_pool(values, size):
    size = tuple(size)
    spatial = values.ndim - 4
    pool = (F.adaptive_avg_pool2d, F.adaptive_avg_pool3d)[spatial]
    if values.device.type != "mps":
        return pool(values, size)
    ratios = [divmod(n, s) for n, s in zip(values.shape[2:], size, strict=True)]
    if any(remainder for _, remainder in ratios):
        return pool(values.cpu(), size).to(values.device)
    kernel = tuple(quotient for quotient, _ in ratios)
    return (F.avg_pool2d, F.avg_pool3d)[spatial](values, kernel)
