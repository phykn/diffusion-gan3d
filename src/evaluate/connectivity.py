import numpy as np
import scipy.ndimage as ndi
import torch

from src.device import wide_dtype
from src.evaluate.label import phase_labels


def transition_counts(previous, current, num_phases: int) -> torch.Tensor:
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


def spanning_fractions(mask: np.ndarray) -> tuple[float, float, float]:
    phase_voxels = int(mask.sum())
    if phase_voxels == 0:
        return 0.0, 0.0, 0.0
    components, _ = ndi.label(mask)
    sizes = np.bincount(components.reshape(-1))
    fractions = []
    for axis in range(3):
        first = np.unique(np.take(components, 0, axis=axis))
        last = np.unique(np.take(components, -1, axis=axis))
        spanning = np.intersect1d(first, last)
        spanning = spanning[spanning != 0]
        fractions.append(float(sizes[spanning].sum() / phase_voxels))
    return tuple(fractions)
