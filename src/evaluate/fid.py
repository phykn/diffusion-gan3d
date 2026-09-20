import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.evaluate.image import prepare_images


def compute_fid(
    real,
    generated,
    device: torch.device | str,
    feature: int = 2048,
) -> float:
    metric = FrechetInceptionDistance(
        feature=feature,
        normalize=False,
    ).to(device)
    metric.update(prepare_images(real, device), real=True)
    metric.update(prepare_images(generated, device), real=False)
    return float(metric.compute().cpu())
