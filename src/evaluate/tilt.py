import numpy as np
import scipy.ndimage as ndi

# Volume axes are (z, y, x); "zy" is the slope dz/dy of a particle's voxels.
TILT_PAIRS = {"zy": (0, 1), "zx": (0, 2), "yx": (1, 2)}


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
