import math

import numpy as np
import taufactor as tau
import torch


def tortuosity(
    volume,
    phase: int = 0,
    axis: int = 0,
    device: torch.device | str | None = None,
    convergence: float = 1e-3,
) -> float:
    """Return TauFactor's diffusion-based tortuosity for one phase and axis."""
    if not isinstance(phase, int) or isinstance(phase, bool) or phase < 0:
        raise ValueError("phase must be a non-negative integer.")
    if not isinstance(axis, int) or isinstance(axis, bool) or axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if not math.isfinite(convergence) or convergence <= 0.0:
        raise ValueError("convergence must be finite and positive.")

    if isinstance(volume, torch.Tensor):
        values = volume.detach().cpu().numpy()
    else:
        values = np.asarray(volume)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("volume must be a non-empty 3D array.")

    conductive = np.moveaxis(values == phase, axis, 0).astype(np.uint8)
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    solver = tau.Solver(conductive, device=selected_device.type)
    value = solver.solve(verbose=False, conv_crit=convergence)
    return float(np.asarray(value).reshape(-1)[0])
