from dataclasses import dataclass

import torch

from src.evaluate.label import phase_fraction
from src.evaluate.tortuosity import tortuosity


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
