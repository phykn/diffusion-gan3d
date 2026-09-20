from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torchmetrics.image.kid import KernelInceptionDistance

from src.evaluate.image import prepare_images


@dataclass(frozen=True)
class KIDScore:
    mean: float
    std: float
    subset_size: int
    subsets: int


@torch.no_grad()
def compute_kid(
    real_images,
    generated_images,
    device: str | torch.device,
    feature=2048,
    *,
    subset_size: int | None = None,
    subsets: int = 100,
    batch_size: int = 16,
    seed: int | None = None,
) -> KIDScore:
    count = min(len(real_images), len(generated_images))
    if count < 2:
        raise ValueError("KID requires at least two real and two generated images.")
    subset_size = min(50, count) if subset_size is None else subset_size
    if type(subset_size) is not int or not 2 <= subset_size <= count:
        raise ValueError(
            "KID subset_size must be between two and the smaller sample count."
        )
    if type(subsets) is not int or subsets < 1:
        raise ValueError("KID subsets must be a positive integer.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("KID batch_size must be a positive integer.")
    device = torch.device(device)
    devices = (
        []
        if device.type != "cuda"
        else [torch.cuda.current_device() if device.index is None else device.index]
    )
    context = (
        torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
    )
    with context:
        if seed is not None:
            # Seed only the devices owned by this evaluation, restoring them on exit.
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
        metric = KernelInceptionDistance(
            feature=feature,
            subsets=subsets,
            subset_size=subset_size,
            normalize=False,
        ).to(device)
        for images, real in ((real_images, True), (generated_images, False)):
            for start in range(0, len(images), batch_size):
                metric.update(
                    prepare_images(images[start : start + batch_size], device),
                    real=real,
                )
        mean, std = metric.compute()
    return KIDScore(float(mean.cpu()), float(std.cpu()), subset_size, subsets)
