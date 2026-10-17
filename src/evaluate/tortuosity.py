import math

import numpy as np
import taufactor as tau
import torch

from src.device import default_device
from src.evaluate.connectivity import spanning_fractions


def tortuosity(
    volume,
    phase: int = 0,
    axis: int = 0,
    device: torch.device | str | None = None,
    convergence: float = 1e-3,
) -> float:
    if isinstance(volume, torch.Tensor):
        values = volume.detach().cpu().numpy()
    else:
        values = np.asarray(volume)

    conductive = np.moveaxis(values == phase, axis, 0)
    # The solver does not converge without a path between the two faces.
    if spanning_fractions(conductive)[0] == 0.0:
        return math.inf
    conductive = conductive.astype(np.uint8)
    selected_device = torch.device(default_device() if device is None else device)
    solver = tau.Solver(conductive, device=str(selected_device))
    solver.solve(verbose=False, conv_crit=convergence)
    if not solver.converged:
        raise RuntimeError("tortuosity solver did not converge.")
    result = float(np.asarray(solver.tau).reshape(-1)[0])
    if math.isnan(result) or result <= 0:
        raise RuntimeError("tortuosity solver returned an invalid result.")
    return result
