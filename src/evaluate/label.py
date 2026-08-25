import torch


def phase_fraction(values, phase: int = 0) -> float:
    labels = torch.as_tensor(values)
    return float((labels == phase).to(torch.float64).mean())


def phase_fractions(values, num_phases: int) -> torch.Tensor:
    labels = torch.as_tensor(values).reshape(-1).to(torch.long)
    counts = torch.bincount(labels, minlength=num_phases)
    return counts.to(torch.float64).div(labels.numel())


def voxel_accuracy(actual, expected) -> float:
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)
    return float((actual == expected).to(torch.float64).mean())
