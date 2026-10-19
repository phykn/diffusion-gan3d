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
            axis=0,
            device=device,
        )
    except (RuntimeError, ValueError, ZeroDivisionError) as exc:
        # Memory exhaustion is a request failure, not a missing metric; MPS
        # reports it as a plain RuntimeError.
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc):
            raise
        tau = None
    return VolumeMetrics(porosity=porosity, tortuosity=tau)
