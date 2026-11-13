import warnings

import numpy as np

PROBE_STRIDE = 16


def pack(
    size,
    big_radius,
    small_radius,
    big_vf,
    small_vf,
    big_elongation,
    radius_gradient=(1.0, 1.0),
    rng=None,
):
    """Pack particles and return a size³ volume without boundary bias.

    Random packing oscillates in density near any boundary, so y and x wrap
    periodically. z also wraps unless radius_gradient varies with height; then
    particles are packed in z padding, clipped at its edge, and cropped away.
    Volume fractions are targets; random packing can jam before reaching them.
    """
    rng = np.random.default_rng() if rng is None else rng
    graded = radius_gradient[0] != radius_gradient[1]
    padding = 0
    if graded:
        long = big_radius * big_elongation ** (2 / 3)
        short = big_radius * big_elongation ** (-1 / 3)
        radius = max(small_radius, long, short) * max(radius_gradient)
        padding = max(size // 4, int(np.ceil(radius)))
    volume = np.zeros((size + 2 * padding, size, size), dtype=np.uint8)
    particles = []
    region = (slice(padding, padding + size), slice(None), slice(None))
    periodic = np.array((not graded, True, True))
    radius_scales = radius_profile(size, padding, radius_gradient)
    for label, vf in ((2, big_vf), (1, small_vf)):
        target = round(vf * size**3)
        placed = place_particles(
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
    size,
    big_radius,
    small_radius,
    big_vf,
    small_vf,
    big_elongation,
    radius_gradient,
):
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


def radius_profile(size, padding, gradient):
    height = np.arange(-padding, size + padding) / max(size - 1, 1)
    height = height.clip(0, 1)
    return gradient[0] + (gradient[1] - gradient[0]) * height


def place_particles(
    volume,
    particles,
    label,
    target_voxels,
    big_radius,
    small_radius,
    elongation,
    radius_scales,
    region,
    periodic,
    rng,
):
    if target_voxels <= 0:
        return 0

    shape = np.asarray(volume.shape)
    bounds = [part.indices(n) for part, n in zip(region, volume.shape, strict=True)]
    lower = np.asarray([start for start, _, _ in bounds])
    upper = np.asarray([stop for _, stop, _ in bounds])
    templates = {
        scale: make_particle_shape(
            label, big_radius * scale, small_radius * scale, elongation
        )
        for scale in np.unique(radius_scales)
    }
    minimum_axes = templates[min(templates)][0]
    available = np.ones(volume.shape, dtype=bool)
    for center, axes in particles:
        invalidate_centers(available, center, minimum_axes, axes, periodic)

    occupied = 0
    for flat in rng.permutation(volume.size):
        if occupied >= target_voxels:
            break
        if not available.flat[flat]:
            continue
        center = np.asarray(np.unravel_index(flat, volume.shape), dtype=np.int32)
        axes, offsets = templates[radius_scales[center[0]]]
        # Test sparse offsets first. Only a full overlap check can accept a center.
        probe = locate(offsets[::PROBE_STRIDE], center, shape, periodic)
        if volume[tuple(probe.T)].any():
            available.flat[flat] = False
            continue
        positions = locate(offsets, center, shape, periodic)
        labels = tuple(positions.T)
        if volume[labels].any():
            available.flat[flat] = False
            continue

        volume[labels] = label
        particles.append((center, axes))
        occupied += int(
            np.count_nonzero(np.all((positions >= lower) & (positions < upper), axis=1))
        )
        invalidate_centers(available, center, minimum_axes, axes, periodic)
    return occupied


def locate(offsets, center, shape, periodic):
    positions = offsets + center
    np.remainder(positions, shape, out=positions, where=periodic)
    if periodic.all():
        return positions
    return positions[((positions >= 0) & (positions < shape)).all(1)]


def make_particle_shape(
    label,
    big_radius,
    small_radius,
    elongation,
):
    if label == 1:
        axes = np.full(3, small_radius, dtype=np.float64)
    else:
        short = big_radius / elongation ** (1.0 / 3.0)
        long = big_radius * elongation ** (2.0 / 3.0)
        axes = np.asarray((long, short, short), dtype=np.float64)
    return axes, make_offsets(axes)


def make_offsets(axes):
    bounds = np.ceil(axes).astype(np.int32)
    grid = [np.arange(-bound, bound + 1, dtype=np.int32) for bound in bounds]
    offsets = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = np.sum((offsets / axes) ** 2, axis=1) <= 1.0 + 1e-12
    return offsets[keep]


def invalidate_centers(
    valid_centers,
    center,
    candidate,
    previous,
    periodic,
):
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
