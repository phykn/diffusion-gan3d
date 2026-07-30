import torch
import torch.nn.functional as F


def labels_to_clean(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    if labels.ndim not in (3, 4):
        raise ValueError("labels must have shape [B, H, W] or [B, D, H, W].")
    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long.")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty.")
    if int(labels.min()) < 0 or int(labels.max()) >= num_phases:
        raise ValueError("labels contain a phase outside num_phases.")
    one_hot = F.one_hot(labels, num_classes=num_phases).movedim(-1, 1)
    return one_hot.to(torch.float32).mul_(2.0).sub_(1.0)


def clean_to_labels(clean: torch.Tensor) -> torch.Tensor:
    if clean.ndim != 5:
        raise ValueError("clean volume must have shape [B, C, D, H, W].")
    return clean.argmax(dim=1)
