import logging
import math

import numpy as np
import scipy.ndimage as ndi
import torch

from src.device import default_device
from src.phase import PORE_PHASE

logger = logging.getLogger(__name__)

# Volume axes are (z, y, x); "zy" is the slope dz/dy of a particle's voxels.
TILT_PAIRS = {"zy": (0, 1), "zx": (0, 2), "yx": (1, 2)}


def spanning_fractions(mask: np.ndarray) -> tuple[float, float, float]:
    """Per axis, the fraction of phase voxels in components touching both faces."""
    phase_voxels = int(mask.sum())
    if phase_voxels == 0:
        return 0.0, 0.0, 0.0
    components, _ = ndi.label(mask)
    sizes = np.bincount(components.reshape(-1))
    fractions = []
    for axis in range(3):
        first = np.unique(np.take(components, 0, axis=axis))
        last = np.unique(np.take(components, -1, axis=axis))
        spanning = np.intersect1d(first, last)
        spanning = spanning[spanning != 0]
        fractions.append(float(sizes[spanning].sum() / phase_voxels))
    return tuple(fractions)


def tortuosity(
    volume,
    phase: int = 0,
    axis: int = 0,
    device: torch.device | str | None = None,
) -> float:
    import taufactor as tau

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
    solver.solve(verbose=False, conv_crit=1e-3)
    if not solver.converged:
        raise RuntimeError("tortuosity solver did not converge.")
    result = float(np.asarray(solver.tau).reshape(-1)[0])
    if math.isnan(result) or result <= 0:
        raise RuntimeError("tortuosity solver returned an invalid result.")
    return result


def pore_tortuosity(volume: torch.Tensor, device: torch.device) -> float | None:
    """Pore-phase tortuosity through the thickness (z); None if the solver fails."""
    try:
        return tortuosity(volume, phase=PORE_PHASE, axis=0, device=device)
    except (RuntimeError, ValueError, ZeroDivisionError) as exc:
        # Memory exhaustion is a request failure, not a missing metric; MPS
        # reports it as a plain RuntimeError.
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc):
            raise
        logger.warning("pore tortuosity unavailable: %s", exc)
        return None


def component_slopes(labels, phase: int, min_voxels: int = 8) -> dict[str, np.ndarray]:
    """Per-particle least-squares slopes cov(a, b) / var(b) for each axis pair."""
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("tilt labels must be a 3D volume.")
    components, count = ndi.label(labels == phase)
    y, x = (axis.ravel().astype(np.float64) for axis in np.indices(labels.shape[1:]))
    # Accumulate coordinate moments one z slice at a time, so memory follows the
    # slice area rather than the volume.
    names = ("n", "z", "y", "x", "zy", "zx", "yx", "yy", "xx")
    sums = dict.fromkeys(names, np.zeros(count + 1))
    for z, plane in enumerate(components):
        flat = plane.ravel()
        inside = flat > 0
        flat, ys, xs = flat[inside], y[inside], x[inside]

        def total(weights=None):
            return np.bincount(flat, weights, minlength=count + 1)

        n, sy, sx = total(), total(ys), total(xs)
        for name, value in (
            ("n", n),
            ("z", z * n),
            ("y", sy),
            ("x", sx),
            ("zy", z * sy),
            ("zx", z * sx),
            ("yx", total(ys * xs)),
            ("yy", total(ys * ys)),
            ("xx", total(xs * xs)),
        ):
            sums[name] = sums[name] + value
    size = sums["n"][1:]
    keep = size >= min_voxels
    means = {name: sums[name][1:][keep] / size[keep] for name in names[1:]}
    slopes = {}
    for name, (a, b) in TILT_PAIRS.items():
        first, second = "zyx"[a], "zyx"[b]
        cov = means[first + second] - means[first] * means[second]
        var = means[second * 2] - means[second] ** 2
        valid = var > 1e-9
        slopes[name] = cov[valid] / var[valid]
    return slopes


def tilt_balance(volumes, phase: int, threshold: float = 0.05, min_voxels: int = 8):
    """Signed tilt statistics of one phase, pooled over volumes.

    `balance` is the fraction tilted positive minus the fraction tilted
    negative; a mirror-symmetric microstructure has balance and mean near zero.
    """
    pooled = {name: [] for name in TILT_PAIRS}
    for volume in volumes:
        for name, values in component_slopes(volume, phase, min_voxels).items():
            pooled[name].append(values)
    result = {}
    for name, parts in pooled.items():
        values = np.concatenate(parts) if parts else np.zeros(0)
        count = len(values)
        positive = float((values > threshold).mean()) if count else 0.0
        negative = float((values < -threshold).mean()) if count else 0.0
        result[name] = {
            "count": count,
            "positive": positive,
            "negative": negative,
            "balance": positive - negative,
            "mean": float(values.mean()) if count else 0.0,
            "sem": float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0,
        }
    return result


def chord_mean(volume: torch.Tensor, phase: int, axis: int) -> float:
    """Mean length of runs of phase along axis, in voxels."""
    lines = np.moveaxis(np.asarray(volume) == phase, axis, -1)
    lines = lines.reshape(-1, lines.shape[-1])
    edges = np.diff(np.pad(lines, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    runs = int((edges == 1).sum())
    return float(lines.sum() / max(runs, 1))


def interface_density(labels: torch.Tensor, dims) -> float:
    """Fraction of neighbour pairs along dims whose labels differ."""
    values = [
        (
            labels.narrow(dim, 1, labels.shape[dim] - 1)
            != labels.narrow(dim, 0, labels.shape[dim] - 1)
        )
        .float()
        .mean()
        for dim in dims
    ]
    return float(torch.stack(values).mean())
