from pathlib import Path

import torch


class SliceStream:
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def next(self) -> torch.Tensor:
        groups = self.dataset.path_groups
        group = groups[int(torch.randint(len(groups), ()).item())]
        indices = torch.randint(len(group), (self.batch_size,))
        return torch.stack([self.dataset[group[int(i)]] for i in indices])


def validate_bank(
    bank: dict[int, torch.Tensor], domains: dict, size: int, phases: int
) -> None:
    if set(bank) != set(domains):
        raise ValueError("LR bank domains must match the training domains.")
    for domain, volumes in bank.items():
        if (
            volumes.ndim != 4
            or volumes.shape[0] < 1
            or tuple(volumes.shape[1:]) != (size, size, size)
        ):
            raise ValueError(
                f"LR bank domain {domain} must contain N,{size},{size},{size} labels."
            )
        if (
            volumes.dtype.is_floating_point
            or volumes.dtype == torch.bool
            or volumes.min() < 0
            or volumes.max() >= phases
        ):
            raise ValueError("LR bank contains invalid phase labels.")


def load_bank(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "diffusion-gan3d.lr-bank.v1":
        raise ValueError("unsupported LR bank format.")
    return payload
