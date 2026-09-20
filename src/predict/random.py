from contextlib import contextmanager

import torch


@contextmanager
def seeded_rng(seed: int | None, device: torch.device):
    if seed is None:
        yield
        return
    devices = []
    if device.type == "cuda":
        index = device.index
        devices = [torch.cuda.current_device() if index is None else index]
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield
