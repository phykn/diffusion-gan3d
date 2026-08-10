import numpy as np
import scipy.ndimage as ndi
import torch

LabelArray = np.ndarray | torch.Tensor
MetricArray = np.ndarray | torch.Tensor
PercolationVector = np.ndarray | torch.Tensor | tuple[float, float, float]

_SIX_CONNECTIVITY = ndi.generate_binary_structure(3, 1)


def transition_counts(
    previous: LabelArray,
    current: LabelArray,
    num_phases: int,
) -> torch.Tensor:
    """Count every ``previous phase -> current phase`` transition."""
    _num_phases(num_phases)
    before = _labels(previous, "previous")
    after = _labels(current, "current")
    if before.shape != after.shape:
        raise ValueError("previous and current labels must have the same shape.")
    if before.device != after.device:
        raise ValueError("previous and current labels must be on the same device.")
    if int(before.max()) >= num_phases or int(after.max()) >= num_phases:
        raise ValueError(f"labels must contain phases from 0 to {num_phases - 1}.")

    counts = torch.zeros(
        num_phases * num_phases,
        dtype=torch.long,
        device=before.device,
    )
    before_flat = before.reshape(-1)
    after_flat = after.reshape(-1)
    for before_chunk, after_chunk in zip(
        before_flat.split(16_000_000),
        after_flat.split(16_000_000),
        strict=True,
    ):
        pairs = before_chunk.to(torch.long) * num_phases
        pairs.add_(after_chunk.to(torch.long))
        counts.add_(torch.bincount(pairs, minlength=num_phases * num_phases))
    return counts.to(torch.float64).reshape(num_phases, num_phases)


def phase_change_rate(counts: MetricArray) -> float:
    """Return the fraction of transitions whose phase changes."""
    matrix = _counts(counts)
    total = matrix.sum()
    return float((total - matrix.diagonal().sum()) / total)


def transition_tv(first: MetricArray, second: MetricArray) -> float:
    """Return total-variation distance between two transition distributions."""
    first_counts, second_counts = _count_pair(first, second)
    first_dist = first_counts / first_counts.sum()
    second_dist = second_counts / second_counts.sum()
    return float(0.5 * (first_dist - second_dist).abs().sum())


def phase_continuation(counts: MetricArray) -> torch.Tensor:
    """Return ``P(next phase = k | current phase = k)`` for every phase."""
    matrix = _counts(counts, allow_empty=True)
    totals = matrix.sum(dim=1)
    missing = torch.nonzero(totals == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(
            "phase continuation requires source support for every phase; "
            f"missing phases: {missing}."
        )
    return matrix.diagonal() / totals


def continuation_delta(first: MetricArray, second: MetricArray) -> float:
    """Return the largest difference among phases supported in both inputs."""
    first_counts, second_counts = _count_pair(first, second)
    first_totals = first_counts.sum(dim=1)
    second_totals = second_counts.sum(dim=1)
    supported = (first_totals > 0) & (second_totals > 0)
    if not bool(supported.any()):
        raise ValueError(
            "continuation delta requires a phase supported in both inputs."
        )
    first_values = first_counts.diagonal()[supported] / first_totals[supported]
    second_values = second_counts.diagonal()[supported] / second_totals[supported]
    return float((first_values - second_values).abs().max())


def continuation_error(
    predicted: MetricArray,
    target: MetricArray,
    phase: int,
) -> float:
    """Return the absolute continuation error for one phase as a fraction."""
    _phase(phase)
    predicted_values = _continuations(predicted, "predicted")
    target_values = _continuations(target, "target", device=predicted_values.device)
    if predicted_values.shape != target_values.shape:
        raise ValueError("predicted and target continuations must have the same shape.")
    if phase >= predicted_values.numel():
        raise ValueError("phase must index an available continuation value.")
    return float((predicted_values[phase] - target_values[phase]).abs())


def percolating_fraction(
    volume: LabelArray,
    phase: int = 1,
    axis: int = 0,
) -> float:
    """Return the fraction of a phase belonging to components spanning one axis."""
    _axis(axis)
    return percolating_fractions(volume, phase)[axis]


def percolating_fractions(
    volume: LabelArray,
    phase: int = 1,
) -> tuple[float, float, float]:
    """Return non-periodic 6-connected spanning fractions for all three axes."""
    return _percolating_fractions(_phase_mask(volume, phase))


def percolation_errors(
    predicted: PercolationVector,
    target: PercolationVector,
) -> tuple[float, float, float]:
    """Return absolute predicted-target percolating-fraction errors by axis."""
    predicted_values = _percolation_values(predicted, "predicted")
    target_values = _percolation_values(
        target,
        "target",
        device=predicted_values.device,
    )
    return tuple(float(value) for value in (predicted_values - target_values).abs())


def percolation_error(
    predicted: PercolationVector, target: PercolationVector
) -> float:
    """Return the mean absolute percolating-fraction error over three axes."""
    errors = percolation_errors(predicted, target)
    return sum(errors) / len(errors)


def _phase_mask(volume: LabelArray, phase: int) -> np.ndarray:
    _phase(phase)
    labels = _labels(volume, "volume")
    if labels.ndim != 3:
        raise ValueError("volume must be a 3D array.")
    mask = (labels == phase).detach().cpu().numpy()
    if any(size < 2 for size in labels.shape):
        raise ValueError("each volume axis must contain at least two voxels.")
    if not bool(mask.any()):
        raise ValueError(f"volume contains no voxels for phase {phase}.")
    return mask


def _percolating_fractions(mask: np.ndarray) -> tuple[float, float, float]:
    phase_voxels = int(mask.sum())
    try:
        components, component_count = ndi.label(
            mask,
            structure=_SIX_CONNECTIVITY,
            output=np.int32,
        )
    except RuntimeError as error:
        raise ValueError(
            "volume has too many components for int32 labeling."
        ) from error
    del mask

    flags = np.zeros(component_count + 1, dtype=np.uint8)
    for axis in range(3):
        first = np.unique(np.take(components, 0, axis=axis))
        last = np.unique(np.take(components, -1, axis=axis))
        spanning = np.intersect1d(first, last, assume_unique=True)
        spanning = spanning[spanning != 0]
        flags[spanning] |= np.uint8(1 << axis)

    totals = np.zeros(3, dtype=np.int64)
    flat = components.reshape(-1)
    for start in range(0, flat.size, 4_000_000):
        bits = flags[flat[start : start + 4_000_000]]
        for axis in range(3):
            totals[axis] += np.count_nonzero(bits & np.uint8(1 << axis))
    return tuple(float(total / phase_voxels) for total in totals)


def _percolation_values(
    values: PercolationVector,
    name: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device=device)
    if result.shape != (3,):
        raise ValueError(f"{name} percolating fractions must contain three axes.")
    if not bool(torch.isfinite(result).all()) or bool(
        ((result < 0) | (result > 1)).any()
    ):
        raise ValueError(f"{name} percolating fractions must be within [0, 1].")
    return result


def _axis(axis: int) -> None:
    if not isinstance(axis, int) or isinstance(axis, bool) or axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")


def _count_pair(
    first: MetricArray,
    second: MetricArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_counts = _counts(first)
    second_counts = _counts(second, device=first_counts.device)
    if first_counts.shape != second_counts.shape:
        raise ValueError("transition count matrices must have the same shape.")
    return first_counts, second_counts


def _counts(
    values: MetricArray,
    *,
    allow_empty: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    counts = torch.as_tensor(values, dtype=torch.float64, device=device)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or not counts.shape[0]:
        raise ValueError("transition counts must be a non-empty square matrix.")
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("transition counts must be finite and non-negative.")
    if not allow_empty and float(counts.sum()) <= 0.0:
        raise ValueError("transition counts must contain at least one transition.")
    return counts


def _continuations(
    values: MetricArray,
    name: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device=device)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} continuations must be a non-empty vector.")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} continuations must be finite.")
    return result


def _labels(values: LabelArray, name: str) -> torch.Tensor:
    labels = torch.as_tensor(values)
    if labels.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if labels.is_complex():
        raise ValueError(f"{name} must contain integer phase labels.")
    if labels.is_floating_point() and (
        not bool(torch.isfinite(labels).all())
        or not bool(torch.equal(labels, labels.round()))
    ):
        raise ValueError(f"{name} must contain finite integer phase labels.")
    if int(labels.min()) < 0:
        raise ValueError(f"{name} must contain non-negative phase labels.")
    return labels


def _phase(phase: int) -> None:
    if not isinstance(phase, int) or isinstance(phase, bool) or phase < 0:
        raise ValueError("phase must be a non-negative integer.")


def _num_phases(num_phases: int) -> None:
    if (
        not isinstance(num_phases, int)
        or isinstance(num_phases, bool)
        or num_phases < 1
    ):
        raise ValueError("num_phases must be a positive integer.")
