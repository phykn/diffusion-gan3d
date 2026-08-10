import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def metric_images(
    images,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert grayscale or RGB images to TorchMetrics' uint8 RGB format."""
    source = np.array(images, copy=True) if isinstance(images, np.ndarray) else images
    values = torch.as_tensor(source).clone()
    if values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4 or values.shape[1] not in (1, 3):
        raise ValueError(
            "images must have shape [N, H, W], [N, 1, H, W], or [N, 3, H, W]."
        )
    if values.numel() == 0:
        raise ValueError("images must not be empty.")
    if values.is_complex() or (
        values.is_floating_point() and not bool(torch.isfinite(values).all())
    ):
        raise ValueError("images must contain finite real values.")

    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0.0:
        raise ValueError("images must contain non-negative values.")
    if maximum <= 1.0:
        values = values.to(torch.float32).mul(255).round().to(torch.uint8)
    elif maximum <= 255.0 and (
        values.dtype == torch.uint8
        or not values.is_floating_point()
        or bool(torch.equal(values, values.round()))
    ):
        values = values.to(torch.uint8)
    else:
        raise ValueError("images must be binary or contain uint8-range intensities.")
    if values.shape[1] == 1:
        values = values.repeat(1, 3, 1, 1)
    return values.to(device) if device is not None else values


def make_kid_metric(
    real,
    device: torch.device | str,
    *,
    feature: int = 2048,
    subsets: int = 100,
    subset_size: int = 50,
) -> KernelInceptionDistance:
    """Create a KID metric that retains the supplied real-image features."""
    metric = KernelInceptionDistance(
        feature=feature,
        subsets=subsets,
        subset_size=subset_size,
        reset_real_features=False,
        normalize=False,
    ).to(device)
    metric.update(metric_images(real, device), real=True)
    return metric


def kid_score(
    metric: KernelInceptionDistance,
    generated,
    device: torch.device | str,
    *,
    seed: int = 0,
) -> tuple[float, float]:
    """Score generated images and clear only the cached generated features."""
    selected_device = torch.device(device)
    cuda_devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            metric.update(metric_images(generated, device), real=False)
            torch.manual_seed(seed)
            if selected_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            mean, std = metric.compute()
            return float(mean.cpu()), float(std.cpu())
        finally:
            metric.reset()


def make_fid_metric(
    real,
    device: torch.device | str,
    *,
    feature: int = 2048,
) -> FrechetInceptionDistance:
    """Create a FID metric that retains the supplied real-image features."""
    metric = FrechetInceptionDistance(
        feature=feature,
        reset_real_features=False,
        normalize=False,
    ).to(device)
    metric.update(metric_images(real, device), real=True)
    return metric


def fid_score(
    metric: FrechetInceptionDistance,
    generated,
    device: torch.device | str,
) -> float:
    """Score generated images and clear only the cached generated features."""
    metric.update(metric_images(generated, device), real=False)
    try:
        return float(metric.compute().cpu())
    finally:
        metric.reset()
