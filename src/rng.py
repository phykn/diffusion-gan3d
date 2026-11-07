"""Run random state: every draw comes from a saved torch.Generator.

Small draws come from CPU generators and move to the device, so a seed gives
the same crops, choices and latents on every device. Diffusion noise volumes
come from a generator on the training device: drawn on the CPU, they took
about 7% of an SR step on MPS. Its state is saved too, so a resume on the
same device type repeats the run exactly.
"""

import hashlib

import numpy as np
import torch

from src.console import warning

# One stream per purpose, so adding draws to one does not shift the others.
STREAMS = ("data", "train", "pairs", "augment", "bank")
NOISE = "noise"


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


class RunRandom:
    def __init__(self, seed: int, device: torch.device | str = "cpu") -> None:
        children = np.random.SeedSequence(seed).spawn(len(STREAMS) + 1)
        seeds = [int(child.generate_state(1, np.uint64)[0] >> 1) for child in children]
        self.generators = {
            name: make_generator(value)
            for name, value in zip(STREAMS, seeds, strict=False)
        }
        self.generators[NOISE] = torch.Generator(device=device).manual_seed(seeds[-1])

    def __getitem__(self, name: str) -> torch.Generator:
        return self.generators[name]

    def state(self) -> dict[str, torch.Tensor]:
        return {name: gen.get_state() for name, gen in self.generators.items()}

    def load_state(self, states: dict[str, torch.Tensor]) -> None:
        if set(states) != set(self.generators):
            raise ValueError("checkpoint random state does not match this version.")
        for name, state in states.items():
            try:
                self.generators[name].set_state(state)
            except RuntimeError:
                if name != NOISE:
                    raise
                # Saved on another device type: only the noise stream leaves
                # the unbroken run, from a seed fixed by the saved state.
                digest = hashlib.sha256(state.numpy().tobytes()).digest()
                self.generators[name].manual_seed(
                    int.from_bytes(digest[:8], "little") >> 1
                )
                warning(
                    "the checkpoint was saved on another device type; "
                    "diffusion noise restarts from a new seed."
                )


def randn(shape, generator, device=None, dtype=torch.float32) -> torch.Tensor:
    values = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return values.to(device=device, dtype=dtype)


def rand(shape, generator, device=None) -> torch.Tensor:
    return torch.rand(tuple(shape), generator=generator).to(device)


def randint(high: int, shape, generator, device=None) -> torch.Tensor:
    return torch.randint(high, tuple(shape), generator=generator).to(device)
