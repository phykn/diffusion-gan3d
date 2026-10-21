import warnings

import numpy as np
from scipy import ndimage

from simul.src.geometry import (
    PROBE_STRIDE,
    invalidate_centers,
    locate,
    radius_profile,
)

PORE, GRAPHITE, SILICON = 0, 1, 2
SHAPE_LIBRARY = 48
FLAKE_WIDTH = (0.6, 1.0)  # in-plane minor/major ratio range of graphite flakes
MAX_PASSES = 8
MIN_PASS_GAIN = 0.01  # stop once a pass adds under 1% of the target
PROBE_BUDGET = 2_000_000  # probe points checked per vectorized batch
COARSE_PROBE = 24  # points in the first, cheap overlap screen


def pack_anode(
    size: int,
    graphite_radius: float,
    graphite_aspect: float,
    graphite_tilt: float,
    graphite_vf,
    silicon_radius: float,
    silicon_vf,
    size_spread: float = 0.0,
    min_section: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pack a battery-anode-like size³ volume: 0 pore, 1 graphite, 2 silicon.

    Graphite flakes are oblate ellipsoids, graphite_aspect thick relative to
    their in-plane radius, lying in the y/x plane with normals tilted from z by
    a normal(0, graphite_tilt) angle in degrees. Silicon spheres fill the pores
    after graphite. A volume fraction is either one number or a list of values
    at evenly spaced heights from z=0 to z=size-1, linearly interpolated along
    z, the electrode thickness; a particle is accepted only while its center
    slice is below that slice's target.
    Radii are lognormal with sigma size_spread. y/x wrap; z is packed with
    padding and cropped so neither face is a hard wall.
    """
    rng = np.random.default_rng() if rng is None else rng
    check_anode(
        size,
        graphite_radius,
        graphite_aspect,
        graphite_tilt,
        graphite_vf,
        silicon_radius,
        silicon_vf,
        size_spread,
        min_section,
    )
    spread = np.exp(3 * size_spread)
    padding = int(np.ceil(graphite_radius * spread))
    volume = np.zeros((size + 2 * padding, size, size), dtype=np.uint8)
    region = (slice(padding, padding + size), slice(None), slice(None))
    periodic = np.array((False, True, True))
    valid = np.ones(volume.shape, dtype=bool)
    inscribed: list[tuple[np.ndarray, float]] = []
    phases = (
        (
            GRAPHITE,
            graphite_vf,
            graphite_shapes(
                graphite_radius,
                graphite_aspect,
                graphite_tilt,
                size_spread,
                min_section,
                rng,
            ),
        ),
        (
            SILICON,
            silicon_vf,
            silicon_shapes(silicon_radius, size_spread, min_section, rng),
        ),
    )
    for label, vf, shapes in phases:
        profile = vf_profile(size, padding, vf)
        fill(volume, valid, inscribed, label, profile * size**2, shapes, periodic, rng)
        reached = (volume[region] == label).mean()
        target = profile[padding : padding + size].mean()
        if reached < target - 0.005:
            name = "graphite" if label == GRAPHITE else "silicon"
            warnings.warn(
                f"{name} jammed at volume fraction {reached:.3f} "
                f"before its {target:.3f} target.",
                stacklevel=2,
            )
    return volume[region].copy()


def vf_profile(size: int, padding: int, vf) -> np.ndarray:
    points = np.atleast_1d(np.asarray(vf, dtype=float))
    height = radius_profile(size, padding, (0.0, 1.0))
    return np.interp(height, np.linspace(0.0, 1.0, len(points)), points)


def depth_profile(geometry: dict) -> dict:
    """Target [pore, graphite, silicon] fractions over z, as a generation vf_profile."""
    vfs = [
        np.atleast_1d(np.asarray(geometry[name], dtype=float))
        for name in ("graphite_vf", "silicon_vf")
    ]
    knots = [np.linspace(0.0, 1.0, len(vf)) for vf in vfs]
    heights = np.union1d(np.union1d(*knots), (0.0, 1.0))
    points = []
    for height in heights:
        graphite, silicon = (
            round(float(np.interp(height, knot, vf)), 6) for knot, vf in zip(knots, vfs)
        )
        pore = round(1.0 - graphite - silicon, 6)
        points.append([round(float(height), 6), [pore, graphite, silicon]])
    return {"axis": "z", "interpolation": "linear", "points": points}


def check_anode(
    size,
    graphite_radius,
    graphite_aspect,
    graphite_tilt,
    graphite_vf,
    silicon_radius,
    silicon_vf,
    size_spread=0.0,
    min_section=0,
) -> None:
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer.")
    if graphite_radius < 1 or silicon_radius < 1:
        raise ValueError("radii must be at least one voxel.")
    if not 0.0 < graphite_aspect <= 1.0:
        raise ValueError("graphite_aspect must be in (0, 1].")
    if not np.isfinite(graphite_tilt) or graphite_tilt < 0:
        raise ValueError("graphite_tilt must be a non-negative angle in degrees.")
    for name, vf in (("graphite_vf", graphite_vf), ("silicon_vf", silicon_vf)):
        value = np.asarray(vf, dtype=float)
        if value.ndim > 1 or value.size == 0 or not ((value >= 0) & (value <= 1)).all():
            raise ValueError(
                f"{name} must be a fraction or a list of fractions in [0, 1]."
            )
    profiles = [vf_profile(size, 0, vf) for vf in (graphite_vf, silicon_vf)]
    if (profiles[0] + profiles[1] > 1.0 + 1e-9).any():
        raise ValueError("graphite_vf and silicon_vf must sum to at most one.")
    if not 0.0 <= size_spread <= 1.0:
        raise ValueError("size_spread must be in [0, 1].")
    if type(min_section) is not int or min_section < 0:
        raise ValueError("min_section must be a non-negative integer.")


def graphite_shapes(radius, aspect, tilt, spread, min_section, rng):
    shapes = []
    for _ in range(SHAPE_LIBRARY):
        major = radius * rng.lognormal(0.0, spread)
        axes = np.array((major * aspect, major, major * rng.uniform(*FLAKE_WIDTH)))
        shapes.append(make_shape(axes, flake_rotation(tilt, rng), min_section))
    return shapes


def silicon_shapes(radius, spread, min_section, rng):
    return [
        make_shape(
            np.full(3, radius * rng.lognormal(0.0, spread)), np.eye(3), min_section
        )
        for _ in range(SHAPE_LIBRARY)
    ]


def flake_rotation(tilt: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate body (z, y, x) so the flake normal leaves z by a random tilt."""
    angle = np.deg2rad(rng.normal(0.0, tilt))
    azimuth, spin = rng.uniform(0.0, 2 * np.pi, 2)
    return about_z(azimuth) @ tilt_toward_x(angle) @ about_z(spin)


def about_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))


def tilt_toward_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))


def make_shape(axes: np.ndarray, rotation: np.ndarray, min_section: int):
    """Return (voxel offsets, inscribed radius) of a rotated ellipsoid.

    Along each axis, every voxel layer keeps only its largest 2D component, and
    layers smaller than min_section voxels are dropped, so axis-aligned slices
    of the particle never show specks from rims, tips, or digital staircases.
    """
    bound = int(np.ceil(axes.max()))
    grid = np.arange(-bound, bound + 1, dtype=np.int32)
    offsets = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    body = offsets @ rotation
    mask = (np.sum((body / axes) ** 2, axis=1) <= 1.0 + 1e-12).reshape((len(grid),) * 3)
    if min_section > 0:
        # Two rounds, since trimming one axis can shrink layers of another.
        for _ in range(2):
            for axis in range(3):
                for layer in np.moveaxis(mask, axis, 0):
                    labels, count = ndimage.label(layer)
                    if count:
                        sizes = np.bincount(labels.ravel())[1:]
                        largest = np.argmax(sizes) + 1
                        keep = sizes[largest - 1] >= min_section
                        layer[...] = (labels == largest) & keep
    # A sphere this size sits inside the digital shape whatever its rotation.
    return offsets[mask.ravel()], max(0.8 * axes.min() - 1.0, 0.0)


def fill(volume, valid, inscribed, label, budget, shapes, periodic, rng):
    """Place random library shapes while each center slice is under budget.

    budget holds the target voxel count of label per z slice. Each pass tries
    one random shape at every valid center in a slice still under budget;
    passes repeat while they add volume, since a center that rejected one
    rotation or size may accept another.
    """
    shape = np.asarray(volume.shape)
    counts = np.zeros(volume.shape[0])
    smallest = min(radius for _, radius in shapes)
    valid[...] = True
    for center, radius in inscribed:
        prune(valid, center, smallest + radius, periodic)
    total = budget.sum()
    for _ in range(MAX_PASSES):
        before = counts.sum()
        open_slices = counts < budget
        if total <= 0 or not open_slices.any():
            break
        candidates = rng.permutation(np.flatnonzero(valid & open_slices[:, None, None]))
        start = 0
        while start < len(candidates):
            offsets, radius = shapes[rng.integers(len(shapes))]
            probe = offsets[::PROBE_STRIDE]
            stop = start + max(64, PROBE_BUDGET // len(probe))
            batch = candidates[start:stop]
            start = stop
            centers = np.stack(np.unravel_index(batch, volume.shape), 1)
            # One vectorized sparse probe rejects most overlapping centers;
            # survivors are rechecked exactly because earlier ones may land.
            keep = valid.flat[batch] & (counts[centers[:, 0]] < budget[centers[:, 0]])
            # A few spread points first, then the denser probe on survivors.
            for points in (probe[:: max(1, len(probe) // COARSE_PROBE)], probe):
                keep[keep] = ~probe_hits(volume, centers[keep], points)
            for center in centers[keep].astype(np.int32):
                if not valid[tuple(center)] or counts[center[0]] >= budget[center[0]]:
                    continue
                positions = locate(offsets, center, shape, periodic)
                indices = tuple(positions.T)
                if volume[indices].any():
                    continue
                volume[indices] = label
                inscribed.append((center, radius))
                counts += np.bincount(positions[:, 0], minlength=len(counts))
                prune(valid, center, smallest + radius, periodic)
        if counts.sum() - before < MIN_PASS_GAIN * total:
            break


def probe_hits(volume, centers, probe) -> np.ndarray:
    """Return which centers' probe points touch an occupied voxel (z clipped)."""
    depth_n, height, width = volume.shape
    depth = centers[:, None, 0] + probe[None, :, 0]
    inside = (depth >= 0) & (depth < depth_n)
    np.clip(depth, 0, depth_n - 1, out=depth)
    rows = (centers[:, None, 1] + probe[None, :, 1]) % height
    cols = (centers[:, None, 2] + probe[None, :, 2]) % width
    occupied = volume.ravel()[(depth * height + rows) * width + cols] != 0
    return (occupied & inside).any(1)


def prune(valid, center, reach, periodic) -> None:
    if reach > 0:
        invalidate_centers(valid, center, np.zeros(3), np.full(3, reach), periodic)
