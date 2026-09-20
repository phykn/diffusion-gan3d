import numpy as np
import scipy.ndimage as ndi
import torch

from src.evaluate.label import phase_labels

_CONNECTIVITY = ndi.generate_binary_structure(3, 1)


def transition_counts(previous, current, num_phases: int) -> torch.Tensor:
    previous = phase_labels(previous, num_phases)
    current = phase_labels(current, num_phases).to(previous.device)
    if previous.shape != current.shape:
        raise ValueError("transition labels must have the same shape.")
    pairs = previous.reshape(-1) * num_phases + current.reshape(-1)
    counts = torch.bincount(pairs, minlength=num_phases * num_phases)
    return counts.reshape(num_phases, num_phases).to(torch.float64)


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
    first = torch.as_tensor(first, dtype=torch.float64)
    second = torch.as_tensor(second, dtype=torch.float64, device=first.device)
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


def percolating_fractions(volume, phase: int = 1) -> tuple[float, float, float]:
    labels = phase_labels(volume)
    if labels.ndim != 3 or any(size < 2 for size in labels.shape):
        raise ValueError("volume must be a 3D array with at least two voxels per axis.")
    mask = (labels == phase).detach().cpu().numpy()
    if not bool(mask.any()):
        raise ValueError(f"volume contains no voxels for phase {phase}.")
    return spanning_fractions(mask)


def spanning_fractions(mask: np.ndarray) -> tuple[float, float, float]:
    phase_voxels = int(mask.sum())
    if phase_voxels == 0:
        return 0.0, 0.0, 0.0
    components, _ = ndi.label(mask, structure=_CONNECTIVITY)
    sizes = np.bincount(components.reshape(-1))
    fractions = []
    for axis in range(3):
        first = np.unique(np.take(components, 0, axis=axis))
        last = np.unique(np.take(components, -1, axis=axis))
        spanning = np.intersect1d(first, last)
        spanning = spanning[spanning != 0]
        fractions.append(float(sizes[spanning].sum() / phase_voxels))
    return tuple(fractions)
