from dataclasses import dataclass

import torch

from ..evaluate import phase_fraction, tortuosity


@dataclass(frozen=True)
class VolumeMetrics:
    porosity: float
    tortuosity: float | None


def measure_volume(
    volume: torch.Tensor,
    device: torch.device,
) -> VolumeMetrics:
    porosity = phase_fraction(volume, phase=0)
    try:
        tau = tortuosity(
            volume,
            phase=0,
            axis=1,
            device=device,
        )
    except (RuntimeError, ValueError, ZeroDivisionError):
        tau = None
    return VolumeMetrics(porosity=porosity, tortuosity=tau)
