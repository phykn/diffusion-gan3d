import numpy as np
import scipy.ndimage as ndi
import torch

_CONNECTIVITY = ndi.generate_binary_structure(3, 1)


def transition_counts(previous, current, num_phases: int) -> torch.Tensor:
    previous = torch.as_tensor(previous, dtype=torch.long)
    current = torch.as_tensor(current, dtype=torch.long, device=previous.device)
    pairs = previous.reshape(-1) * num_phases + current.reshape(-1)
    if bool((pairs < 0).any()) or bool(
        (pairs >= num_phases * num_phases).any()
    ):
        raise ValueError(f"labels must contain phases from 0 to {num_phases - 1}.")
    counts = torch.bincount(pairs, minlength=num_phases * num_phases)
    return counts.reshape(num_phases, num_phases).to(torch.float64)


def transition_tv(first, second) -> float:
    first = torch.as_tensor(first, dtype=torch.float64)
    second = torch.as_tensor(second, dtype=torch.float64, device=first.device)
    first = first / first.sum()
    second = second / second.sum()
    return float(0.5 * (first - second).abs().sum())


def continuation_delta(first, second) -> float:
    first = torch.as_tensor(first, dtype=torch.float64)
    second = torch.as_tensor(second, dtype=torch.float64, device=first.device)
    first_totals = first.sum(dim=1)
    second_totals = second.sum(dim=1)
    supported = (first_totals > 0) & (second_totals > 0)
    first_values = first.diagonal()[supported] / first_totals[supported]
    second_values = second.diagonal()[supported] / second_totals[supported]
    return float((first_values - second_values).abs().max())


def percolating_fractions(volume, phase: int = 1) -> tuple[float, float, float]:
    labels = torch.as_tensor(volume)
    if labels.ndim != 3 or any(size < 2 for size in labels.shape):
        raise ValueError("volume must be a 3D array with at least two voxels per axis.")
    mask = (labels == phase).detach().cpu().numpy()
    if not bool(mask.any()):
        raise ValueError(f"volume contains no voxels for phase {phase}.")
    phase_voxels = int(mask.sum())
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
