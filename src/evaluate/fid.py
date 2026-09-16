import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def prepare_fid_images(
    images,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    values = torch.as_tensor(images)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.dtype != torch.uint8 or int(values.max()) <= 1:
        values = values.to(torch.float32).mul(255).round().to(torch.uint8)
    if values.shape[1] == 1:
        values = values.repeat(1, 3, 1, 1)
    return values.to(device) if device is not None else values


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
    metric.update(prepare_fid_images(real, device), real=True)
    metric.update(prepare_fid_images(generated, device), real=False)
    return float(metric.compute().cpu())
