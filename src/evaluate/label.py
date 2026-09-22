import torch

from src.plane import AXES

LABEL_CHUNK_ELEMENTS = 1024**2


def _label_chunks(values):
    # Split views before conversion, including for non-contiguous input. A
    # flatten/reshape of the full input could itself allocate another volume.
    pending = [values]
    while pending:
        part = pending.pop()
        if part.numel() <= LABEL_CHUNK_ELEMENTS:
            yield phase_labels(part)
        else:
            axis = max(range(part.ndim), key=lambda axis: part.shape[axis])
            pending.extend(part.split((part.shape[axis] + 1) // 2, dim=axis))


def phase_labels(values, num_phases: int | None = None) -> torch.Tensor:
    values = torch.as_tensor(values)
    if values.numel() == 0 or values.is_complex():
        raise ValueError("phase labels must be non-empty real integers.")
    if num_phases is not None and (type(num_phases) is not int or num_phases < 1):
        raise ValueError("num_phases must be a positive integer.")
    labels = values.to(torch.int64)
    valid = (labels >= 0).all()
    if values.is_floating_point():
        valid &= (values == labels.to(values.dtype)).all()
    message = "phase labels must be non-negative integers."
    if num_phases is not None:
        valid &= (labels < num_phases).all()
        message = f"labels must contain integer phases from 0 to {num_phases - 1}."
    if labels.device.type == "cuda":
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)
    return labels


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
        values = phase_labels(images, num_phases)
        counts.append(torch.bincount(values.flatten(), minlength=num_phases).float())
        total += values.numel()
    return torch.stack(counts).sum(0).div(total)


def phase_fraction(values, phase: int = 0) -> float:
    if type(phase) is not int or phase < 0:
        raise ValueError("phase must be a non-negative integer.")
    values = torch.as_tensor(values)
    count = torch.zeros((), dtype=torch.int64, device=values.device)
    for labels in _label_chunks(values):
        count += (labels == phase).sum()
    return count.item() / values.numel()


def phase_fractions(values, num_phases: int) -> torch.Tensor:
    labels = phase_labels(values, num_phases).reshape(-1)
    counts = torch.bincount(labels, minlength=num_phases)
    return counts.to(torch.float64).div(labels.numel())


def voxel_accuracy(actual, expected) -> float:
    actual = phase_labels(actual)
    expected = phase_labels(expected).to(actual.device)
    if actual.shape != expected.shape:
        raise ValueError("actual and expected labels must have the same shape.")
    return float((actual == expected).to(torch.float64).mean())
