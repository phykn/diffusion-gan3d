import torch

from src.device import assert_all


def encode_phases(labels, num_phases):
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("phase labels must have an integer dtype.")
    labels = labels.to(torch.int64)
    in_range = ((labels >= 0) & (labels < num_phases)).all()
    assert_all(in_range, f"phase labels must be in [0, {num_phases - 1}].")
    batch, *spatial = labels.shape
    fractions = torch.zeros(
        (batch, num_phases, *spatial), device=labels.device, dtype=torch.float32
    )
    return fractions.scatter_(1, labels.unsqueeze(1), 1.0)
