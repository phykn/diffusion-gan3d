from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Particle:
    center: tuple[int, int, int]
    axes: tuple[float, float, float]
    label: int


def pack(
    *,
    size: int,
    big_radius: int,
    small_radius: int,
    big_fraction: float,
    small_fraction: float,
    big_elongation: float,
) -> np.ndarray:
    small_r = float(small_radius)
    # Packing outside the retained field reduces boundary-biased particle fractions.
    work = size * 3 // 2
    vol = np.zeros((work,) * 3, dtype=np.uint8)
    ids = np.full((work,) * 3, -1, dtype=np.int32)
    parts: list[Particle] = []
    target = {
        1: round(small_fraction * vol.size),
        2: round(big_fraction * vol.size),
    }

    for label in (2, 1):
        _place(
            vol=vol,
            ids=ids,
            parts=parts,
            label=label,
            target=target[label],
            big_r=float(big_radius),
            small_r=small_r,
            elong=big_elongation,
        )

    return _crop(vol, size)


def _place(
    *,
    vol: np.ndarray,
    ids: np.ndarray,
    parts: list[Particle],
    label: int,
    target: int,
    big_r: float,
    small_r: float,
    elong: float,
) -> None:
    axes, offsets = _make_particle(
        label=label,
        big_r=big_r,
        small_r=small_r,
        elong=elong,
    )
    count = round(target / len(offsets))
    if count <= 0:
        return

    valid = _find_centers(vol.shape[0], offsets)
    for part in parts:
        _invalidate_centers(
            valid,
            np.asarray(part.center),
            axes,
            np.asarray(part.axes),
        )

    order = np.random.permutation(np.flatnonzero(valid))
    placed = 0
    for flat in order:
        if placed >= count:
            break
        if not valid.flat[flat]:
            continue
        center = np.asarray(
            np.unravel_index(flat, valid.shape),
            dtype=np.int32,
        )
        positions = offsets + center
        key = tuple(positions.T)
        if np.any(ids[key] >= 0):
            valid.flat[flat] = False
            continue

        idx = len(parts)
        vol[key] = label
        ids[key] = idx
        parts.append(
            Particle(
                center=tuple(int(value) for value in center),
                axes=tuple(float(value) for value in axes),
                label=label,
            )
        )
        placed += 1
        _invalidate_centers(valid, center, axes, axes)


def _crop(
    vol: np.ndarray,
    size: int,
) -> np.ndarray:
    low = (vol.shape[0] - size) // 2
    high = low + size
    key = (slice(low, high),) * 3
    return vol[key].copy()


def _make_particle(
    *,
    label: int,
    big_r: float,
    small_r: float,
    elong: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = big_r if label == 2 else small_r
    if label == 1:
        axes = np.full(3, radius, dtype=np.float64)
    else:
        short = radius / elong ** (1.0 / 3.0)
        long = radius * elong ** (2.0 / 3.0)
        axes = np.asarray((long, short, short), dtype=np.float64)
    return axes, _make_offsets(axes)


def _make_offsets(axes: np.ndarray) -> np.ndarray:
    bounds = np.ceil(axes).astype(np.int32)
    z, y, x = np.meshgrid(
        np.arange(-bounds[0], bounds[0] + 1, dtype=np.int32),
        np.arange(-bounds[1], bounds[1] + 1, dtype=np.int32),
        np.arange(-bounds[2], bounds[2] + 1, dtype=np.int32),
        indexing="ij",
    )
    offsets = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
    keep = np.sum((offsets / axes) ** 2, axis=1) <= 1.0 + 1e-12
    return offsets[keep]


def _find_centers(size: int, offsets: np.ndarray) -> np.ndarray:
    valid = np.zeros((size,) * 3, dtype=bool)
    low = -offsets.min(axis=0)
    high = size - 1 - offsets.max(axis=0)
    if np.any(low > high):
        return valid
    valid[
        low[0] : high[0] + 1,
        low[1] : high[1] + 1,
        low[2] : high[2] + 1,
    ] = True
    return valid


def _invalidate_centers(
    valid: np.ndarray,
    center: np.ndarray,
    candidate: np.ndarray,
    previous: np.ndarray,
) -> None:
    span = candidate + previous
    bounds = np.ceil(span).astype(np.int32)
    low = np.maximum(center - bounds, 0)
    high = np.minimum(center + bounds + 1, valid.shape)
    z, y, x = np.ogrid[
        low[0] - center[0] : high[0] - center[0],
        low[1] - center[1] : high[1] - center[1],
        low[2] - center[2] : high[2] - center[2],
    ]
    hit = (z / span[0]) ** 2 + (y / span[1]) ** 2 + (x / span[2]) ** 2 <= 1.0
    view = valid[
        low[0] : high[0],
        low[1] : high[1],
        low[2] : high[2],
    ]
    view[hit] = False
