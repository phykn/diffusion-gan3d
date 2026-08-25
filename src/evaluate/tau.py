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
    if isinstance(volume, torch.Tensor):
        values = volume.detach().cpu().numpy()
    else:
        values = np.asarray(volume)

    conductive = np.moveaxis(values == phase, axis, 0).astype(np.uint8)
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    solver = tau.Solver(conductive, device=selected_device.type)
    value = solver.solve(verbose=False, conv_crit=convergence)
    return float(np.asarray(value).reshape(-1)[0])
