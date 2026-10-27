import warnings
from dataclasses import dataclass

import numpy as np

PROBE_STRIDE = 16


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
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pack particles and return a size³ volume without boundary bias.

    Random packing oscillates in density near any boundary, so y and x wrap
    periodically. z also wraps unless radius_gradient varies with height; then
    particles are packed in z padding, clipped at its edge, and cropped away.
    Volume fractions are targets; random packing can jam before reaching them.
    """
    rng = np.random.default_rng() if rng is None else rng
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
    graded = radius_gradient[0] != radius_gradient[1]
    padding = (
        max(size // 4, int(np.ceil(largest_radius * max(radius_gradient))))
        if graded
        else 0
    )
    volume = np.zeros((size + 2 * padding, size, size), dtype=np.uint8)
    particles: list[Particle] = []
    region = (slice(padding, padding + size), slice(None), slice(None))
    periodic = np.array((not graded, True, True))
    radius_scales = radius_profile(size, padding, radius_gradient)
    for label, vf in ((2, big_vf), (1, small_vf)):
        target = round(vf * size**3)
        placed = place(
            volume=volume,
            particles=particles,
            label=label,
            target_voxels=target,
            big_radius=float(big_radius),
            small_radius=float(small_radius),
            elongation=big_elongation,
            radius_scales=radius_scales,
            region=region,
            periodic=periodic,
            rng=rng,
        )
        if placed < target:
            warnings.warn(
                f"phase {label} jammed at volume fraction {placed / size**3:.3f} "
                f"before its {vf} target.",
                stacklevel=2,
            )
    return volume[region].copy()


def check_geometry(
    size: int,
    big_radius: int,
    small_radius: int,
    big_vf: float,
    small_vf: float,
    big_elongation: float,
    radius_gradient: tuple[float, float],
) -> None:
    if type(size) is not int or size < 1:
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
    particles: list[Particle],
    label: int,
    target_voxels: int,
    big_radius: float,
    small_radius: float,
    elongation: float,
    radius_scales: np.ndarray,
    region: tuple[slice, slice, slice] | None = None,
    periodic=(False, False, False),
    rng: np.random.Generator | None = None,
) -> int:
    """Place particles until region holds target_voxels of label; return the count.

    Particles wrap around periodic axes and are clipped at the other edges.
    A voxel is occupied once it holds any nonzero label.
    """
    if target_voxels <= 0:
        return 0
    rng = np.random.default_rng() if rng is None else rng
    shape = np.asarray(volume.shape)
    region = (slice(None),) * 3 if region is None else region
    bounds = [part.indices(n) for part, n in zip(region, volume.shape, strict=True)]
    low = np.asarray([start for start, _, _ in bounds])
    high = np.asarray([stop for _, stop, _ in bounds])
    periodic = np.asarray(periodic, dtype=bool)
    shapes = {
        scale: make_particle_shape(
            label, big_radius * scale, small_radius * scale, elongation
        )
        for scale in np.unique(radius_scales)
    }
    minimum_axes = shapes[min(shapes)][0]
    valid_centers = np.ones(volume.shape, dtype=bool)
    for particle in particles:
        invalidate_centers(
            valid_centers,
            np.asarray(particle.center),
            minimum_axes,
            np.asarray(particle.axes),
            periodic,
        )

    placed = 0
    for flat in rng.permutation(volume.size):
        if placed >= target_voxels:
            break
        if not valid_centers.flat[flat]:
            continue
        center = np.asarray(np.unravel_index(flat, volume.shape), dtype=np.int32)
        axes, offsets = shapes[radius_scales[center[0]]]
        # A sparse probe rejects most overlapping candidates cheaply; the full
        # check below still decides every acceptance exactly.
        probe = locate(offsets[::PROBE_STRIDE], center, shape, periodic)
        if volume[tuple(probe.T)].any():
            valid_centers.flat[flat] = False
            continue
        positions = locate(offsets, center, shape, periodic)
        indices = tuple(positions.T)
        if volume[indices].any():
            valid_centers.flat[flat] = False
            continue

        volume[indices] = label
        particles.append(
            Particle(
                center=tuple(int(value) for value in center),
                axes=tuple(float(value) for value in axes),
            )
        )
        placed += int(((positions >= low) & (positions < high)).all(1).sum())
        # Conservative candidate pruning; exact voxel overlap is still checked
        # above because subsequent particles can have different radii.
        invalidate_centers(valid_centers, center, minimum_axes, axes, periodic)
    return placed


def locate(offsets, center, shape, periodic) -> np.ndarray:
    positions = offsets + center
    np.remainder(positions, shape, out=positions, where=periodic)
    if periodic.all():
        return positions
    return positions[((positions >= 0) & (positions < shape)).all(1)]


def make_particle_shape(
    label: int,
    big_radius: float,
    small_radius: float,
    elongation: float,
) -> tuple[np.ndarray, np.ndarray]:
    if label == 1:
        axes = np.full(3, small_radius, dtype=np.float64)
    else:
        short = big_radius / elongation ** (1.0 / 3.0)
        long = big_radius * elongation ** (2.0 / 3.0)
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
    periodic,
) -> None:
    span = candidate + previous
    bounds = np.ceil(span).astype(np.int32)
    offsets, indices = [], []
    for axis, size in enumerate(valid_centers.shape):
        offset = np.arange(-bounds[axis], bounds[axis] + 1)
        index = center[axis] + offset
        if periodic[axis]:
            index %= size
        else:
            keep = (index >= 0) & (index < size)
            offset, index = offset[keep], index[keep]
        offsets.append(offset)
        indices.append(index)
    z, y, x = np.ix_(*offsets)
    hit = np.nonzero((z / span[0]) ** 2 + (y / span[1]) ** 2 + (x / span[2]) ** 2 <= 1)
    valid_centers[tuple(index[part] for index, part in zip(indices, hit))] = False
