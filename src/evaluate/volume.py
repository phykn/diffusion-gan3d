import torch

from src.evaluate.tortuosity import tortuosity
from src.phase import PORE_PHASE


def pore_tortuosity(volume: torch.Tensor, device: torch.device) -> float | None:
    """Pore-phase tortuosity through the thickness (z); None if the solver fails."""
    try:
        return tortuosity(volume, phase=PORE_PHASE, axis=0, device=device)
    except (RuntimeError, ValueError, ZeroDivisionError) as exc:
        # Memory exhaustion is a request failure, not a missing metric; MPS
        # reports it as a plain RuntimeError.
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc):
            raise
        return None
