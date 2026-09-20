import torch

from src.evaluate.label import phase_labels
from src.prepare.profile import rebin_profile


def phase_profile(values, phases, bins=None):
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


def compare_profiles(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("profiles must describe the same height bins and phases.")
    error = prediction.float() - target.to(prediction)
    return {
        "mae": error.abs().mean(),
        "rmse": error.square().mean().sqrt(),
        "max_error": error.abs().max(),
    }
