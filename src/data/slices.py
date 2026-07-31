import torch
import torch.nn.functional as F


def sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    axis: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous.shape != current.shape:
        raise ValueError("previous and current volumes must have the same shape.")
    if previous.ndim != 5:
        raise ValueError("volumes must have shape [B, C, D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if count <= 0:
        raise ValueError("count must be positive.")

    batch_idx = torch.randint(
        previous.shape[0],
        (count,),
        device=previous.device,
    )
    plane_idx = torch.randint(
        previous.shape[axis + 2],
        (count,),
        device=previous.device,
    )
    previous = previous.movedim(axis + 2, 2)
    current = current.movedim(axis + 2, 2)
    return previous[batch_idx, :, plane_idx], current[batch_idx, :, plane_idx]


def encode_labels(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
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
