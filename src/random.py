from contextlib import contextmanager

import torch


@contextmanager
def seeded_rng(seed: int | None, device: torch.device):
    """Temporarily seed CPU and the selected device, restoring both on exit."""
    if seed is None:
        yield
        return
    devices = []
    if device.type == "cuda":
        index = device.index
        devices = [torch.cuda.current_device() if index is None else index]
    elif device.type == "mps":
        devices = [0]
    device_type = "mps" if device.type == "mps" else "cuda"
    with torch.random.fork_rng(devices=devices, device_type=device_type):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        if device.type == "cuda":
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        yield
