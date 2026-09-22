import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.evaluate.image import prepare_images


@torch.no_grad()
def compute_fid(
    real,
    generated,
    device: torch.device | str,
    feature: int | torch.nn.Module = 2048,
    *,
    batch_size: int = 16,
) -> float:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("FID batch_size must be a positive integer.")
    if min(len(real), len(generated)) < 2:
        raise ValueError("FID requires at least two real and two generated images.")
    metric = FrechetInceptionDistance(
        feature=feature,
        normalize=False,
    ).to(device)
    for images, is_real in ((real, True), (generated, False)):
        for start in range(0, len(images), batch_size):
            metric.update(
                prepare_images(images[start : start + batch_size], device),
                real=is_real,
            )
    return float(metric.compute().cpu())
