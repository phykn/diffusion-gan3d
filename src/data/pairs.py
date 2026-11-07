"""Draw sections of generated volumes for the plane critics."""

import torch


def crop_images(
    images: torch.Tensor, size: int | tuple[int, int], generator: torch.Generator
) -> torch.Tensor:
    crop_h, crop_w = (size, size) if isinstance(size, int) else size
    if crop_h < 1 or crop_w < 1:
        raise ValueError("crop size must be a positive integer.")
    height, width = images.shape[-2:]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size must fit inside the images.")
    if (height, width) == (crop_h, crop_w):
        return images
    top = torch.randint(height - crop_h + 1, (images.shape[0],), generator=generator)
    left = torch.randint(width - crop_w + 1, (images.shape[0],), generator=generator)
    return torch.stack(
        [
            image[..., row : row + crop_h, col : col + crop_w]
            for image, row, col in zip(images, top.tolist(), left.tolist(), strict=True)
        ]
    )


def sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    axis: int,
    count: int,
    crop_shape: int | tuple[int, int],
    generator: torch.Generator,
    height: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """count sections normal to axis from [B, C, D, H, W] volume pairs.

    Returns (previous, current[, height]) sections cropped to crop_shape.
    """
    size = previous.shape[axis + 2]
    choices = torch.randint(previous.shape[0] * size, (count,), generator=generator)
    choices = choices.to(previous.device)
    batch_indices, plane_indices = choices // size, choices % size
    previous = previous.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    current = current.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    channels = previous.shape[1]
    values = (previous, current)
    if height is not None:
        values += (height.movedim(axis + 2, 2)[batch_indices, :, plane_indices],)
    pairs = crop_images(torch.cat(values, dim=1), crop_shape, generator)
    result = (pairs[:, :channels], pairs[:, channels : 2 * channels])
    return result if height is None else (*result, pairs[:, 2 * channels :].detach())
