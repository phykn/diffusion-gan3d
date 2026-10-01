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


def validate_bank_height(
    bank: dict, origins: dict | None, extents: dict, crop_size: int
) -> dict[int, torch.Tensor]:
    """Validate source-pixel coordinates and resolve one extent per bank sample."""
    if origins is None or set(origins) != set(bank):
        raise ValueError("height-conditioned SR requires bank crop origins.")
    if set(extents) != set(bank):
        raise ValueError("LR bank height extent domains must match the bank.")
    resolved = {}
    for domain, volumes in bank.items():
        origin = origins[domain]
        extent = extents[domain]
        if extent is None:
            raise ValueError("SR bank requires per-sample height extents.")
        extent = torch.as_tensor(extent)
        if (
            extent.shape not in ((), (len(volumes),))
            or not torch.isfinite(extent).all()
            or (extent <= 0).any()
        ):
            raise ValueError("invalid LR bank height extents.")
        if (
            origin.shape != (len(volumes),)
            or not torch.isfinite(origin).all()
            or (origin < 0).any()
            or (origin + crop_size > extent).any()
        ):
            raise ValueError("invalid LR bank crop origins.")
        resolved[domain] = extent.expand(len(volumes)) if extent.ndim == 0 else extent
    return resolved


def load_bank(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "diffusion-gan3d.lr-bank":
        raise ValueError("unsupported LR bank format.")
    return payload
