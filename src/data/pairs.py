import torch


def sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    axis: int,
    count: int,
    generator: torch.Generator,
    height: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """count sections normal to axis from [B, C, D, H, W] volume pairs:
    (previous, current[, height])."""
    size = previous.shape[axis + 2]
    choices = torch.randint(previous.shape[0] * size, (count,), generator=generator)
    choices = choices.to(previous.device)
    batch_indices, plane_indices = choices // size, choices % size
    previous = previous.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    current = current.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    if height is None:
        return previous, current
    height = height.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    return previous, current, height.detach()
