from dataclasses import dataclass

import torch
from torchmetrics.image.kid import KernelInceptionDistance

from src.evaluate.image import check_batch_size, metric_device, update_image_metric
from src.random import seeded_rng


@dataclass(frozen=True)
class KIDScore:
    mean: float
    std: float
    subset_size: int
    subsets: int


@torch.no_grad()
def compute_kid(
    real,
    generated,
    device: torch.device | str,
    feature: int | torch.nn.Module = 2048,
    *,
    subset_size: int | None = None,
    subsets: int = 100,
    batch_size: int = 16,
    seed: int | None = None,
) -> KIDScore:
    count = min(len(real), len(generated))
    if count < 2:
        raise ValueError("KID requires at least two real and two generated images.")
    subset_size = min(50, count) if subset_size is None else subset_size
    if type(subset_size) is not int or not 2 <= subset_size <= count:
        raise ValueError(
            "KID subset_size must be between two and the smaller sample count."
        )
    if type(subsets) is not int or subsets < 1:
        raise ValueError("KID subsets must be a positive integer.")
    check_batch_size(batch_size)
    device = metric_device(device)
    with seeded_rng(seed, device):
        metric = KernelInceptionDistance(
            feature=feature,
            subsets=subsets,
            subset_size=subset_size,
            normalize=False,
        ).to(device)
        update_image_metric(metric, real, generated, device, batch_size)
        mean, std = metric.compute()
    return KIDScore(float(mean.cpu()), float(std.cpu()), subset_size, subsets)
