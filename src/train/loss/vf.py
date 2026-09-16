import torch

from src import AXES


def compute_vf(
    batches: dict[int, torch.Tensor],
    num_phases: int,
) -> torch.Tensor:
    if not batches or not set(batches).issubset(AXES):
        raise ValueError("batches must contain at least one valid axis.")
    if any(
        not isinstance(images, torch.Tensor) or images.ndim not in (3, 4)
        for images in batches.values()
    ):
        raise ValueError("training crops must have shape [B,H,W] or [B,C,H,W].")
    counts = []
    total = 0
    for images in batches.values():
        if images.numel() == 0:
            raise ValueError("training crops must not be empty.")
        if images.ndim == 4:
            if images.shape[1] != num_phases:
                raise ValueError("phase channels must match num_phases.")
            counts.append(images.float().sum(dim=(0, 2, 3)))
            total += images.numel() // num_phases
            continue
        values = images.to(torch.long)
        valid = ((values >= 0) & (values < num_phases)).all()
        if values.device.type == "cuda":
            torch._assert_async(
                valid, "training images contain a phase outside num_phases."
            )
        elif not bool(valid):
            raise ValueError("training images contain a phase outside num_phases.")
        counts.append(torch.bincount(values.flatten(), minlength=num_phases).float())
        total += values.numel()
    return torch.stack(counts).sum(0).div(total)


def compute_vf_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    present = present.to(probs.device)
    predicted = probs.to(torch.float32).mean(dim=(2, 3, 4))
    target = target.to(torch.float32)
    target_log = torch.where(
        target > 0.0,
        target.log(),
        torch.zeros_like(target),
    )
    predicted_log = predicted.clamp_min(1e-6).log()
    per_sample = (target * (target_log - predicted_log)).sum(dim=1)
    return (per_sample * present).sum() / present.sum().clamp_min(1)
