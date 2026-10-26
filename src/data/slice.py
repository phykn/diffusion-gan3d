import torch

from src.data.augment import crop_images


def sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    axis: int,
    count: int,
    crop_shape: int | tuple[int, int],
    height: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    size = previous.shape[axis + 2]
    # Draw on the CPU generator so seeded runs match across devices.
    choices = torch.randint(previous.shape[0] * size, (count,)).to(previous.device)
    batch_indices, plane_indices = choices // size, choices % size
    previous = previous.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    current = current.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    channels = previous.shape[1]
    values = (previous, current)
    if height is not None:
        values += (height.movedim(axis + 2, 2)[batch_indices, :, plane_indices],)
    pairs = crop_images(torch.cat(values, dim=1), crop_shape)
    result = (pairs[:, :channels], pairs[:, channels : 2 * channels])
    return result if height is None else (*result, pairs[:, 2 * channels :].detach())
