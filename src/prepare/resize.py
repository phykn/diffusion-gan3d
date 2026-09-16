import math

import torch
import torch.nn.functional as F


def scaled_size(size: int, scale: float) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale < 1
    ):
        raise ValueError("scale_factor must be finite and at least 1.")
    value = size * scale
    if not math.isfinite(value) or not math.isclose(
        value, round(value), abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError(
            "size * scale_factor must be an integer; sizes are never rounded."
        )
    return round(value)


def phase_channels(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    if labels.dtype.is_floating_point or labels.dtype == torch.bool:
        raise ValueError("phase labels must have an integer dtype.")
    if labels.numel() == 0:
        raise ValueError(f"phase labels must be in [0, {num_phases - 1}].")
    valid = ((labels >= 0) & (labels < num_phases)).all()
    if labels.device.type == "cuda":
        torch._assert_async(valid, "phase labels are outside num_phases.")
    elif not bool(valid):
        raise ValueError(f"phase labels must be in [0, {num_phases - 1}].")
    return F.one_hot(labels.long(), num_phases).movedim(-1, 1).float()


def resize_phases(probs: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Resize B,C,H,W or B,C,D,H,W phase channels, never numeric label IDs."""
    if tuple(probs.shape[2:]) == tuple(shape):
        return probs
    if all(new <= old for new, old in zip(shape, probs.shape[2:], strict=True)):
        return F.interpolate(probs, size=shape, mode="area")
    mode = "trilinear" if probs.ndim == 5 else "bilinear"
    return F.interpolate(probs, size=shape, mode=mode, align_corners=False)


def resize_labels(labels: torch.Tensor, size: int, num_phases: int) -> torch.Tensor:
    probs = phase_channels(labels.unsqueeze(0), num_phases)
    return resize_phases(probs, (size,) * labels.ndim).argmax(1).squeeze(0)


def resize_crop(
    labels: torch.Tensor,
    size: int,
    num_phases: int,
) -> torch.Tensor:
    """Return C,H,W phase fractions without discarding subpixel phase occupancy."""
    probs = phase_channels(labels.unsqueeze(0), num_phases)
    return resize_phases(probs, (size,) * labels.ndim).squeeze(0)


def downsample(probs: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Area-average phase occupancy, matching the fractional coarse contract."""
    return resize_phases(probs.float(), shape)
