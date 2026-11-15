import logging

import torch
from fastapi.responses import Response

from src.axes import PORE_PHASE
from src.stats.volume import tortuosity

logger = logging.getLogger(__name__)


def volume_response(volume, device, tortuosity):
    shape = ",".join(str(axis) for axis in volume.shape)
    headers = {"X-Volume-Shape": shape}
    if tortuosity:
        tau = pore_tortuosity(volume, device=device)
        headers["X-Tortuosity"] = "unavailable" if tau is None else f"{tau:.8g}"
    return Response(
        volume.numpy().tobytes(),
        media_type="application/octet-stream",
        headers=headers,
    )


def pore_tortuosity(volume, device):
    try:
        return tortuosity(volume, phase=PORE_PHASE, axis=0, device=device)
    except (RuntimeError, ValueError, ZeroDivisionError) as exc:
        # Memory exhaustion is a request failure, not a missing metric; MPS
        # reports it as a plain RuntimeError.
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc):
            raise
        logger.warning("pore tortuosity unavailable: %s", exc)
        return None
