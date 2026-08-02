import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Particle:
    center: tuple[int, int, int]
    axes: tuple[float, float, float]


def pack(
    size: int,
    big_radius: int,
    small_radius: int,
    big_vf: float,
    small_vf: float,
    big_elongation: float,
) -> np.ndarray:
    check_geometry(
        size,
        big_radius,
        small_radius,
        big_vf,
        small_vf,
        big_elongation,
    )
    # Packing outside the retained field reduces boundary-biased particle VF values.
    work_size = size * 3 // 2
    volume = np.zeros((work_size,) * 3, dtype=np.uint8)
    occupied = np.zeros_like(volume, dtype=bool)
    particles: list[Particle] = []
    target_voxels = {
        1: round(small_vf * volume.size),
        2: round(big_vf * volume.size),
    }

    for label in (2, 1):
        place(
            volume=volume,
            occupied=occupied,
            particles=particles,
            label=label,
            target_voxels=target_voxels[label],
            big_radius=float(big_radius),
            small_radius=float(small_radius),
            elongation=big_elongation,
        )

    return crop(volume, size)


def check_geometry(
    size: int,
    big_radius: int,
    small_radius: int,
    big_vf: float,
    small_vf: float,
    big_elongation: float,
) -> None:
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer.")
    for name, radius in (
        ("big_radius", big_radius),
        ("small_radius", small_radius),
    ):
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, vf in (("big_vf", big_vf), ("small_vf", small_vf)):
        if (
            not isinstance(vf, (int, float))
            or isinstance(vf, bool)
            or not math.isfinite(vf)
            or not 0.0 <= vf <= 1.0
        ):
            raise ValueError(f"{name} must be between zero and one.")
    if big_vf + small_vf > 1.0:
        raise ValueError("big_vf and small_vf must sum to at most one.")
    if (
        not isinstance(big_elongation, (int, float))
        or isinstance(big_elongation, bool)
        or not math.isfinite(big_elongation)
        or big_elongation <= 0.0
    ):
        raise ValueError("big_elongation must be finite and positive.")


def place(
    volume: np.ndarray,
    occupied: np.ndarray,
    particles: list[Particle],
    label: int,
    target_voxels: int,
    big_radius: float,
    small_radius: float,
    elongation: float,
) -> None:
    axes, offsets = make_particle_shape(
        label=label,
        big_radius=big_radius,
        small_radius=small_radius,
        elongation=elongation,
    )
    particle_count = round(target_voxels / len(offsets))
    if particle_count <= 0:
        return

    valid_centers = find_valid_centers(volume.shape[0], offsets)
    for particle in particles:
        invalidate_centers(
            valid_centers,
            np.asarray(particle.center),
            axes,
            np.asarray(particle.axes),
        )

    order = np.random.permutation(np.flatnonzero(valid_centers))
    placed = 0
    for flat in order:
        if placed >= particle_count:
            break
        if not valid_centers.flat[flat]:
            continue
        center = np.asarray(
            np.unravel_index(flat, valid_centers.shape),
            dtype=np.int32,
        )
        positions = offsets + center
        indices = tuple(positions.T)
        if np.any(occupied[indices]):
            valid_centers.flat[flat] = False
            continue

        volume[indices] = label
        occupied[indices] = True
        particles.append(
            Particle(
                center=tuple(int(value) for value in center),
                axes=tuple(float(value) for value in axes),
            )
        )
        placed += 1
        invalidate_centers(valid_centers, center, axes, axes)


def crop(
    volume: np.ndarray,
    size: int,
) -> np.ndarray:
    low = (volume.shape[0] - size) // 2
    high = low + size
    region = (slice(low, high),) * 3
    return volume[region].copy()


def make_particle_shape(
    label: int,
    big_radius: float,
    small_radius: float,
    elongation: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = big_radius if label == 2 else small_radius
    if label == 1:
        axes = np.full(3, radius, dtype=np.float64)
    else:
        short = radius / elongation ** (1.0 / 3.0)
        long = radius * elongation ** (2.0 / 3.0)
        axes = np.asarray((long, short, short), dtype=np.float64)
    return axes, make_offsets(axes)


def make_offsets(axes: np.ndarray) -> np.ndarray:
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


def find_valid_centers(size: int, offsets: np.ndarray) -> np.ndarray:
    valid_centers = np.zeros((size,) * 3, dtype=bool)
    low = -offsets.min(axis=0)
    high = size - 1 - offsets.max(axis=0)
    if np.any(low > high):
        return valid_centers
    valid_centers[
        low[0] : high[0] + 1,
        low[1] : high[1] + 1,
        low[2] : high[2] + 1,
    ] = True
    return valid_centers


def invalidate_centers(
    valid_centers: np.ndarray,
    center: np.ndarray,
    candidate: np.ndarray,
    previous: np.ndarray,
) -> None:
    span = candidate + previous
    bounds = np.ceil(span).astype(np.int32)
    low = np.maximum(center - bounds, 0)
    high = np.minimum(center + bounds + 1, valid_centers.shape)
    z, y, x = np.ogrid[
        low[0] - center[0] : high[0] - center[0],
        low[1] - center[1] : high[1] - center[1],
        low[2] - center[2] : high[2] - center[2],
    ]
    hit = (z / span[0]) ** 2 + (y / span[1]) ** 2 + (x / span[2]) ** 2 <= 1.0
    view = valid_centers[
        low[0] : high[0],
        low[1] : high[1],
        low[2] : high[2],
    ]
    view[hit] = False
