import hashlib

import numpy as np
import torch

from src.console import warning

# One stream per purpose, so adding draws to one does not shift the others.
STREAMS = ("data", "train", "pairs", "augment", "bank")
NOISE = "noise"


def make_generator(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed))


class RunRandom:
    def __init__(self, seed, device="cpu"):
        names = (*STREAMS, NOISE)
        children = np.random.SeedSequence(seed).spawn(len(names))
        self.generators = {
            name: torch.Generator(
                device=device if name == NOISE else "cpu"
            ).manual_seed(int(child.generate_state(1, np.uint64)[0] >> 1))
            for name, child in zip(names, children, strict=True)
        }

    def __getitem__(self, name):
        return self.generators[name]

    def state(self):
        return {name: gen.get_state() for name, gen in self.generators.items()}

    def load_state(self, states):
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


def randn(shape, generator, device=None, dtype=torch.float32):
    values = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return values.to(device=device, dtype=dtype)


def rand(shape, generator, device=None):
    return torch.rand(tuple(shape), generator=generator).to(device)


def randint(high, shape, generator, device=None):
    return torch.randint(high, tuple(shape), generator=generator).to(device)
