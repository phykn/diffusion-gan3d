from contextlib import contextmanager

import torch


@contextmanager
def preserved_rng(device: torch.device):
    """Restore CPU and selected-device random state when the block exits."""
    device_type = "mps" if device.type == "mps" else "cuda"
    with torch.random.fork_rng(
        devices=_device_indices(device), device_type=device_type
    ):
        yield


@contextmanager
def seeded_rng(seed: int | None, device: torch.device):
    """Temporarily seed CPU and the selected device, restoring both on exit."""
    if seed is None:
        yield
        return
    with preserved_rng(device):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        if device.type == "cuda":
            with torch.cuda.device(_device_indices(device)[0]):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        yield


def _device_indices(device: torch.device) -> list[int]:
    if device.type == "cuda":
        return [torch.cuda.current_device() if device.index is None else device.index]
    return [0] if device.type == "mps" else []
