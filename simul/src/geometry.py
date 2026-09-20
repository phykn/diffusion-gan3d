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
    radius_gradient: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    check_geometry(
        size,
        big_radius,
        small_radius,
        big_vf,
        small_vf,
        big_elongation,
        radius_gradient,
    )
    largest_radius = max(
        small_radius,
        big_radius * max(big_elongation ** (2 / 3), big_elongation ** (-1 / 3)),
    )
    padding = max(size // 4, int(np.ceil(largest_radius * max(radius_gradient))))
    work_size = size + 2 * padding
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
            radius_scales=radius_profile(size, padding, radius_gradient),
        )

    low = (volume.shape[0] - size) // 2
    region = (slice(low, low + size),) * 3
    return volume[region].copy()


def check_geometry(
    size: int,
    big_radius: int,
    small_radius: int,
    big_vf: float,
    small_vf: float,
    big_elongation: float,
    radius_gradient: tuple[float, float] = (1.0, 1.0),
) -> None:
    if size < 1:
        raise ValueError("size must be a positive integer.")
    if big_radius < 1 or small_radius < 1:
        raise ValueError("radii must be positive.")
    if not 0.0 <= big_vf <= 1.0 or not 0.0 <= small_vf <= 1.0:
        raise ValueError("volume fractions must be between zero and one.")
    if big_vf + small_vf > 1.0:
        raise ValueError("big_vf and small_vf must sum to at most one.")
    if not np.isfinite(big_elongation) or big_elongation <= 0.0:
        raise ValueError("big_elongation must be positive.")
    gradient = np.asarray(radius_gradient, dtype=float)
    if (
        gradient.shape != (2,)
        or not np.isfinite(gradient).all()
        or (gradient <= 0).any()
    ):
        raise ValueError("radius_gradient must contain two finite positive scales.")


def radius_profile(size: int, padding: int, gradient) -> np.ndarray:
    height = np.clip((np.arange(size + 2 * padding) - padding) / max(size - 1, 1), 0, 1)
    return gradient[0] + (gradient[1] - gradient[0]) * height


def place(
    volume: np.ndarray,
    occupied: np.ndarray,
    particles: list[Particle],
    label: int,
    target_voxels: int,
    big_radius: float,
    small_radius: float,
    elongation: float,
    radius_scales: np.ndarray,
) -> None:
    if target_voxels <= 0:
        return
    shapes = {
        scale: make_particle_shape(
            label, big_radius * scale, small_radius * scale, elongation
        )
        for scale in np.unique(radius_scales)
    }
    minimum_axes = shapes[min(shapes)][0]
    # Each candidate's bounds use its local radius, including the z extent.
    valid_centers = np.zeros(volume.shape, dtype=bool)
    for z, scale in enumerate(radius_scales):
        offsets = shapes[scale][1]
        low, high = -offsets.min(axis=0), np.asarray(volume.shape) - offsets.max(axis=0)
        if low[0] <= z < high[0]:
            valid_centers[z, low[1] : high[1], low[2] : high[2]] = True
    for particle in particles:
        invalidate_centers(
            valid_centers,
            np.asarray(particle.center),
            minimum_axes,
            np.asarray(particle.axes),
        )

    order = np.random.permutation(np.flatnonzero(valid_centers))
    placed = 0
    for flat in order:
        if placed >= target_voxels:
            break
        if not valid_centers.flat[flat]:
            continue
        center = np.asarray(
            np.unravel_index(flat, valid_centers.shape),
            dtype=np.int32,
        )
        axes, offsets = shapes[radius_scales[center[0]]]
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
        placed += len(offsets)
        # Conservative candidate pruning; exact voxel overlap is still checked
        # above because subsequent particles can have different radii.
        invalidate_centers(valid_centers, center, minimum_axes, axes)


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
