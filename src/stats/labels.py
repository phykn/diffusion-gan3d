import torch

from src.data.resize import rebin_profile
from src.device import assert_all, wide_dtype


def phase_labels(values, num_phases=None):
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
    assert_all(valid, message)
    return labels


def phase_fractions(values, num_phases):
    values = torch.as_tensor(values)
    counts = None
    for labels in _label_chunks(values, num_phases):
        chunk_counts = torch.bincount(labels.reshape(-1), minlength=num_phases)
        counts = chunk_counts if counts is None else counts.add_(chunk_counts)
    return counts.to(wide_dtype(counts.device)).div(values.numel())


def _label_chunks(values, num_phases):
    # Split views before conversion, including for non-contiguous input. A
    # flatten/reshape of the full input could itself allocate another volume.
    pending = [values]
    while pending:
        part = pending.pop()
        if part.numel() <= 1024**2:
            yield phase_labels(part, num_phases)
        else:
            axis = max(range(part.ndim), key=lambda axis: part.shape[axis])
            pending.extend(part.split((part.shape[axis] + 1) // 2, dim=axis))


def voxel_accuracy(actual, expected):
    actual = phase_labels(actual)
    expected = phase_labels(expected).to(actual.device)
    if actual.shape != expected.shape:
        raise ValueError("actual and expected labels must have the same shape.")
    return float((actual == expected).to(wide_dtype(actual.device)).mean())


def phase_profile(values, phases, bins=None):
    """Phase fractions per z of [B, K, D, H, W] fractions or [B, D, H, W] labels."""
    if values.ndim == 5:
        if values.shape[1] != phases:
            raise ValueError("phase count differs from the probability channels.")
        result = values.float().mean((-1, -2))
    elif values.ndim == 4:
        labels = phase_labels(values, phases)
        result = torch.stack(
            [(labels == phase).float().mean((-1, -2)) for phase in range(phases)], 1
        )
    else:
        raise ValueError("expected batched 3D labels or phase fractions.")
    return result if bins is None else rebin_profile(result, bins)
