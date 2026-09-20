from pathlib import Path

import torch


def validate_bank(
    bank: dict[int, torch.Tensor], domains: dict, size: int, phases: int
) -> dict[int, torch.Tensor]:
    if set(bank) != set(domains):
        raise ValueError("LR bank domains must match the training domains.")
    for domain, volumes in bank.items():
        if (
            volumes.ndim != 5
            or volumes.shape[0] < 1
            or tuple(volumes.shape[1:]) != (phases, size, size, size)
        ):
            raise ValueError(
                f"LR bank domain {domain} must contain N,{phases},{size},{size},{size} fractions."
            )
        if (
            not volumes.dtype.is_floating_point
            or not torch.isfinite(volumes).all()
            or volumes.min() < 0
            or volumes.max() > 1
            or not torch.allclose(
                volumes.sum(1), torch.ones_like(volumes[:, 0]), atol=1e-5
            )
        ):
            raise ValueError("LR bank contains invalid phase fractions.")
    return bank


def load_bank(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "diffusion-gan3d.lr-bank":
        raise ValueError("unsupported LR bank format.")
    return payload
