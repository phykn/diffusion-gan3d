from fastapi.responses import Response

from src.stats.volume import pore_tortuosity


def volume_response(volume, device, tortuosity=False):
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
