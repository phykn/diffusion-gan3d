from dataclasses import dataclass

import torch

from ..evaluate import phase_fraction, tortuosity

TORTUOSITY_AXIS = 1


@dataclass(frozen=True)
class VolumeMetrics:
    porosity: float
    tortuosity: float | None


def measure_volume(
    volume: torch.Tensor,
    *,
    device: torch.device,
) -> VolumeMetrics:
    """Measure phase-0 porosity and axis-1 diffusive tortuosity."""
    porosity = phase_fraction(volume, phase=0)
    try:
        tau = tortuosity(
            volume,
            phase=0,
            axis=TORTUOSITY_AXIS,
            device=device,
        )
    except (RuntimeError, ValueError, ZeroDivisionError):
        tau = None
    return VolumeMetrics(porosity=porosity, tortuosity=tau)
