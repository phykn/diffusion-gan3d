import torch


def sample_volume_slices(
    volume: torch.Tensor,
    *,
    axis: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if volume.ndim != 5:
        raise ValueError("volume must have shape [B, C, D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if count <= 0:
        raise ValueError("count must be positive.")
    batch = volume.shape[0]
    size = volume.shape[axis + 2]
    volume_indices = torch.randint(
        batch,
        (count,),
        device=volume.device,
    )
    plane_indices = torch.randint(
        size,
        (count,),
        device=volume.device,
    )
    selected = []
    for volume_index, plane_index in zip(
        volume_indices.tolist(),
        plane_indices.tolist(),
        strict=True,
    ):
        item = volume[volume_index]
        selected.append(item.select(axis + 1, plane_index))
    return torch.stack(selected), plane_indices


def sample_volume_pair_slices(
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

    batch = previous.shape[0]
    size = previous.shape[axis + 2]
    volume_indices = torch.randint(
        batch,
        (count,),
        device=previous.device,
    )
    plane_indices = torch.randint(
        size,
        (count,),
        device=previous.device,
    )
    previous_slices = []
    current_slices = []
    for volume_index, plane_index in zip(
        volume_indices.tolist(),
        plane_indices.tolist(),
        strict=True,
    ):
        previous_slices.append(
            previous[volume_index].select(axis + 1, plane_index)
        )
        current_slices.append(current[volume_index].select(axis + 1, plane_index))
    return torch.stack(previous_slices), torch.stack(current_slices)
