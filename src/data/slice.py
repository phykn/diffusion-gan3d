import torch

from src.anchor import AnchorCondition
from src.data.augment import crop_images
from src.plane import AXES


def sample_slices(volume: torch.Tensor, axis: int, count: int) -> torch.Tensor:
    planes = volume.movedim(axis + 2, 1)
    planes = planes.flatten(0, 1)
    indices = torch.randint(planes.shape[0], (count,), device=volume.device)
    return planes[indices]


def sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    axis: int,
    count: int,
    crop_shape: int | tuple[int, int],
    anchor: AnchorCondition | None = None,
    measured: AnchorCondition | None = None,
    height: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    size = previous.shape[axis + 2]
    excluded = (
        set()
        if measured is None
        else {r.index for r in measured.regions if r.axis == axis}
    )
    active = (
        ()
        if measured is None
        else (
            range(previous.shape[0])
            if measured.active_batches is None
            else measured.active_batches
        )
    )
    candidates = [
        (batch, index)
        for batch in range(previous.shape[0])
        for index in range(size)
        if batch not in active or index not in excluded
    ]
    if not candidates:
        shape = previous.movedim(axis + 2, 2).shape
        empty = previous.new_empty((0, shape[1], *shape[-2:]))
        return (empty, empty) if height is None else (empty, empty, empty[:, :1])
    choices = torch.randint(len(candidates), (count,)).tolist()
    selected = [candidates[i] for i in choices]
    centers = []
    if anchor is not None:
        axes = [normal for normal in AXES if normal != axis]
        focused = []
        for region in anchor.regions:
            if region.axis == axis:
                continue
            other = [normal for normal in AXES if normal != region.axis]
            start = region.row if other[0] == axis else region.col
            length = region.height if other[0] == axis else region.width
            focused.extend(
                (batch, index, region)
                for batch, index in candidates
                if (anchor.active_batches is None or batch in anchor.active_batches)
                and start <= index < start + length
            )
        for slot in range(max(1, count // 2) if focused else 0):
            batch, index, region = focused[int(torch.randint(len(focused), ()))]
            coordinates = {region.axis: region.index}
            other = [normal for normal in AXES if normal != region.axis]
            coordinates[other[0]] = region.row + int(torch.randint(region.height, ()))
            coordinates[other[1]] = region.col + int(torch.randint(region.width, ()))
            selected[slot] = (batch, index)
            centers.append(tuple(coordinates[normal] for normal in axes))
    batch_indices, plane_indices = zip(*selected)
    batch_indices = torch.tensor(batch_indices, device=previous.device)
    plane_indices = torch.tensor(plane_indices, device=previous.device)
    previous = previous.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    current = current.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    channels = previous.shape[1]
    values = (previous, current)
    if height is not None:
        values += (height.movedim(axis + 2, 2)[batch_indices, :, plane_indices],)
    pairs = crop_images(
        torch.cat(values, dim=1),
        crop_shape,
        centers,
    )
    result = (pairs[:, :channels], pairs[:, channels : 2 * channels])
    return result if height is None else (*result, pairs[:, 2 * channels :].detach())


def _relation_indices(
    index: int,
    size: int,
    gap: int,
) -> tuple[int, int, int] | None:
    if gap < 1:
        return None
    if index - gap >= 0 and index + gap < size:
        return index - gap, index, index + gap
    if index + 2 * gap < size:
        return index, index + gap, index + 2 * gap
    if index - 2 * gap >= 0:
        return index - 2 * gap, index - gap, index
    return None
