import torch


def prepare_images(images, device: torch.device | str | None = None) -> torch.Tensor:
    values = torch.as_tensor(images)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] not in (1, 3) or values.numel() == 0:
        raise ValueError("images must have non-empty shape N,H,W or N,1|3,H,W.")
    if values.dtype == torch.bool:
        values = values.to(torch.uint8).mul(255)
    elif values.is_floating_point():
        if not bool((torch.isfinite(values) & (values >= 0) & (values <= 1)).all()):
            raise ValueError("floating-point images must be finite and in [0,1].")
        values = values.to(torch.float32).mul(255).round().to(torch.uint8)
    elif values.dtype != torch.uint8:
        raise ValueError("images must use bool, floating-point or uint8 values.")
    if values.shape[1] == 1:
        values = values.repeat(1, 3, 1, 1)
    return values.to(device) if device is not None else values
