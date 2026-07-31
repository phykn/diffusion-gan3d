from dataclasses import dataclass

import numpy as np

from .config import GeometryConfig


@dataclass(frozen=True)
class Particle:
    center: tuple[int, int, int]
    axes: tuple[float, float, float]
    label: int


@dataclass(frozen=True)
class PackingReport:
    requested_fractions: tuple[float, float, float]
    achieved_fractions: tuple[float, float, float]
    particle_counts: tuple[int, int]
    phase_contact_counts: tuple[int, int, int]
    particle_contacts: int

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_fractions": list(self.requested_fractions),
            "achieved_fractions": list(self.achieved_fractions),
            "particle_counts": {
                "small": self.particle_counts[0],
                "big": self.particle_counts[1],
            },
            "face_contacts": {
                "background_small": self.phase_contact_counts[0],
                "background_big": self.phase_contact_counts[1],
                "small_big": self.phase_contact_counts[2],
                "particle_pairs": self.particle_contacts,
            },
        }


@dataclass(frozen=True)
class Geometry:
    labels: np.ndarray
    instances: np.ndarray
    particles: tuple[Particle, ...]
    report: PackingReport


def pack(cfg: GeometryConfig) -> Geometry:
    small_r = float(cfg.small_radius)
    # Packing outside the retained field reduces boundary-biased particle fractions.
    work = cfg.size * 3 // 2
    vol = np.zeros((work,) * 3, dtype=np.uint8)
    ids = np.full((work,) * 3, -1, dtype=np.int32)
    parts: list[Particle] = []
    target = {
        1: round(cfg.small_fraction * vol.size),
        2: round(cfg.big_fraction * vol.size),
    }

    for label in (2, 1):
        _place(
            vol=vol,
            ids=ids,
            parts=parts,
            label=label,
            target=target[label],
            big_r=float(cfg.big_radius),
            small_r=small_r,
            elong=cfg.big_elongation,
        )

    vol, ids, parts = _crop(vol, ids, parts, cfg.size)
    counts = np.bincount(vol.ravel(), minlength=3)
    got = tuple(float(value / vol.size) for value in counts)
    want = (
        1.0 - cfg.small_fraction - cfg.big_fraction,
        cfg.small_fraction,
        cfg.big_fraction,
    )
    report = PackingReport(
        requested_fractions=want,
        achieved_fractions=got,
        particle_counts=(
            sum(part.label == 1 for part in parts),
            sum(part.label == 2 for part in parts),
        ),
        phase_contact_counts=_count_phase_contacts(vol),
        particle_contacts=_count_particle_contacts(ids),
    )
    return Geometry(
        labels=vol,
        instances=ids,
        particles=tuple(parts),
        report=report,
    )


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
    ids: np.ndarray,
    parts: list[Particle],
    size: int,
) -> tuple[np.ndarray, np.ndarray, list[Particle]]:
    low = (vol.shape[0] - size) // 2
    high = low + size
    key = (slice(low, high),) * 3
    vol = vol[key].copy()
    ids = ids[key].copy()

    used = np.unique(ids[ids >= 0])
    lookup = np.full(len(parts), -1, dtype=np.int32)
    lookup[used] = np.arange(len(used), dtype=np.int32)
    mask = ids >= 0
    ids[mask] = lookup[ids[mask]]

    shift = np.full(3, low)
    kept = [
        Particle(
            center=tuple(
                int(value) for value in np.asarray(parts[index].center) - shift
            ),
            axes=parts[index].axes,
            label=parts[index].label,
        )
        for index in used
    ]
    return vol, ids, kept


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


def _count_phase_contacts(vol: np.ndarray) -> tuple[int, int, int]:
    pairs = ((0, 1), (0, 2), (1, 2))
    counts = [0, 0, 0]
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        a = vol[tuple(first)]
        b = vol[tuple(second)]
        for index, (left, right) in enumerate(pairs):
            counts[index] += int(
                np.count_nonzero(
                    ((a == left) & (b == right)) | ((a == right) & (b == left))
                )
            )
    return tuple(counts)


def _count_particle_contacts(ids: np.ndarray) -> int:
    hits: set[tuple[int, int]] = set()
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        a = ids[tuple(first)]
        b = ids[tuple(second)]
        mask = (a >= 0) & (b >= 0) & (a != b)
        for left, right in zip(a[mask], b[mask], strict=True):
            hits.add(tuple(sorted((int(left), int(right)))))
    return len(hits)
