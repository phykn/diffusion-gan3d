import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.evaluate.image import check_batch_size, metric_device, update_image_metric


@torch.no_grad()
def compute_fid(
    real,
    generated,
    device: torch.device | str,
    feature: int | torch.nn.Module = 2048,
    *,
    batch_size: int = 16,
) -> float:
    check_batch_size(batch_size)
    if min(len(real), len(generated)) < 2:
        raise ValueError("FID requires at least two real and two generated images.")
    device = metric_device(device)
    metric = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
    update_image_metric(metric, real, generated, device, batch_size)
    return float(metric.compute().cpu())
