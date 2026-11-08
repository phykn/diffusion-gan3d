import torch

from src.data.height import rebin_profile
from src.device import assert_all, wide_dtype

LABEL_CHUNK_ELEMENTS = 1024**2


def _label_chunks(values, num_phases=None):
    # Split views before conversion, including for non-contiguous input. A
    # flatten/reshape of the full input could itself allocate another volume.
    pending = [values]
    while pending:
        part = pending.pop()
        if part.numel() <= LABEL_CHUNK_ELEMENTS:
            yield phase_labels(part, num_phases)
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
    assert_all(valid, message)
    return labels


def phase_fractions(values, num_phases: int) -> torch.Tensor:
    values = torch.as_tensor(values)
    counts = None
    for labels in _label_chunks(values, num_phases):
        chunk_counts = torch.bincount(labels.reshape(-1), minlength=num_phases)
        if counts is None:
            counts = chunk_counts
        else:
            counts.add_(chunk_counts)
    return counts.to(wide_dtype(counts.device)).div(values.numel())


def voxel_accuracy(actual, expected) -> float:
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


def transition_counts(previous, current, num_phases: int) -> torch.Tensor:
    """Counts of label pairs between two adjacent sections, [K, K]."""
    previous = phase_labels(previous, num_phases)
    current = phase_labels(current, num_phases).to(previous.device)
    if previous.shape != current.shape:
        raise ValueError("transition labels must have the same shape.")
    pairs = previous.reshape(-1) * num_phases + current.reshape(-1)
    counts = torch.bincount(pairs, minlength=num_phases * num_phases)
    return counts.reshape(num_phases, num_phases).to(wide_dtype(counts.device))


def transition_tv(first, second) -> float:
    first, second = _transition_matrices(first, second)
    first = first / first.sum()
    second = second / second.sum()
    return float(0.5 * (first - second).abs().sum())


def continuation_delta(first, second) -> float | None:
    first, second = _transition_matrices(first, second)
    first_totals = first.sum(dim=1)
    second_totals = second.sum(dim=1)
    supported = (first_totals > 0) & (second_totals > 0)
    if not bool(supported.any()):
        return None
    first_values = first.diagonal()[supported] / first_totals[supported]
    second_values = second.diagonal()[supported] / second_totals[supported]
    return float((first_values - second_values).abs().max())


def _transition_matrices(first, second) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.as_tensor(first)
    first = first.to(wide_dtype(first.device))
    second = torch.as_tensor(second, dtype=first.dtype, device=first.device)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or first.shape != second.shape
        or first.numel() == 0
    ):
        raise ValueError(
            "transition matrices must have the same non-empty square shape."
        )
    for counts in (first, second):
        if not bool((torch.isfinite(counts) & (counts >= 0)).all()) or not bool(
            counts.sum() > 0
        ):
            raise ValueError(
                "transition counts must be finite, non-negative and nonzero."
            )
    return first, second
