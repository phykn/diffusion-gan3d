import torch

from .. import AXES


def compute_vf(
    batches: dict[int, torch.Tensor],
    num_phases: int,
) -> torch.Tensor:
    if not batches or not set(batches).issubset(AXES):
        raise ValueError("batches must contain at least one valid axis.")
    if any(
        not isinstance(images, torch.Tensor) or images.ndim != 3
        for images in batches.values()
    ):
        raise ValueError("training crops must have shape [B, H, W].")
    labels = []
    for images in batches.values():
        if images.numel() == 0:
            raise ValueError("training crops must not be empty.")
        values = images.to(torch.long)
        lower, upper = torch.aminmax(values)
        if int(lower) < 0 or int(upper) >= num_phases:
            raise ValueError("training images contain a phase outside num_phases.")
        labels.append(values.flatten())
    labels = torch.cat(labels)
    counts = torch.bincount(labels, minlength=num_phases)
    return counts.to(torch.float32).div(labels.numel())


def compute_vf_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    if not bool(present.any()):
        return probs.new_zeros(())
    predicted = probs.to(torch.float32).mean(dim=(2, 3, 4))
    target = target.to(torch.float32)
    target_log = torch.where(
        target > 0.0,
        target.log(),
        torch.zeros_like(target),
    )
    predicted_log = predicted.clamp_min(1e-6).log()
    per_sample = (target * (target_log - predicted_log)).sum(dim=1)
    return per_sample[present].mean()
