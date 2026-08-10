import torch


def phase_fraction(values, phase: int = 0) -> float:
    """Return the fraction of labels equal to ``phase``."""
    labels = _labels(values, "values")
    _phase(phase)
    return float((labels == phase).to(torch.float64).mean())


def phase_fractions(values, num_phases: int) -> torch.Tensor:
    """Return one fraction per phase as a float64 tensor."""
    labels = _labels(values, "values")
    _num_phases(num_phases)
    if int(labels.max()) >= num_phases:
        raise ValueError(f"values must contain phases from 0 to {num_phases - 1}.")

    counts = torch.zeros(num_phases, dtype=torch.long, device=labels.device)
    for chunk in labels.reshape(-1).split(16_000_000):
        counts.add_(
            torch.bincount(
                chunk.to(torch.long),
                minlength=num_phases,
            )
        )
    fractions = counts.to(torch.float64)
    return fractions / fractions.sum()


def voxel_accuracy(actual, expected) -> float:
    """Return the fraction of labels that match at identical coordinates."""
    actual_labels, expected_labels = _label_pair(actual, expected)
    return float((actual_labels == expected_labels).to(torch.float64).mean())


def phase_iou(
    actual,
    expected,
    num_phases: int,
) -> tuple[float, ...]:
    """Return intersection over union for every phase."""
    actual_labels, expected_labels = _label_pair(actual, expected)
    _num_phases(num_phases)
    _validate_phases(actual_labels, expected_labels, num_phases)
    return tuple(
        _overlap_score(actual_labels, expected_labels, phase, "iou")
        for phase in range(num_phases)
    )


def phase_recall(
    actual,
    expected,
    num_phases: int,
) -> tuple[float, ...]:
    """Return the recovered fraction of every expected phase."""
    actual_labels, expected_labels = _label_pair(actual, expected)
    _num_phases(num_phases)
    _validate_phases(actual_labels, expected_labels, num_phases)
    return tuple(
        _overlap_score(actual_labels, expected_labels, phase, "recall")
        for phase in range(num_phases)
    )


def _overlap_score(
    actual: torch.Tensor,
    expected: torch.Tensor,
    phase: int,
    metric: str,
) -> float:
    predicted = actual == phase
    target = expected == phase
    intersection = int((predicted & target).sum())
    if metric == "iou":
        denominator = int((predicted | target).sum())
    else:
        denominator = int(target.sum())
    return 1.0 if denominator == 0 else intersection / denominator


def _label_pair(
    actual,
    expected,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_labels = _labels(actual, "actual")
    expected_labels = _labels(expected, "expected")
    if actual_labels.shape != expected_labels.shape:
        raise ValueError("actual and expected labels must have the same shape.")
    if actual_labels.device != expected_labels.device:
        raise ValueError("actual and expected labels must be on the same device.")
    return actual_labels, expected_labels


def _validate_phases(
    actual: torch.Tensor,
    expected: torch.Tensor,
    num_phases: int,
) -> None:
    if int(actual.max()) >= num_phases or int(expected.max()) >= num_phases:
        raise ValueError(
            f"actual and expected must contain phases from 0 to {num_phases - 1}."
        )


def _labels(values, name: str) -> torch.Tensor:
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
