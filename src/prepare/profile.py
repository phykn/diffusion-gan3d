import torch
import torch.nn.functional as F


def rebin_profile(values, size):
    if type(size) is not int or size < 1 or values.ndim != 3 or values.numel() == 0:
        raise ValueError("profile requires non-empty [B,K,D] and a positive bin count.")
    values = values.float()
    depth = values.shape[-1]
    if size == depth:
        return values
    edges = torch.linspace(
        0, depth, size + 1, device=values.device, dtype=torch.float64
    )
    indices = edges.long().clamp_max(depth - 1)
    # Double precision avoids cancellation when subtracting nearby large integrals.
    cumulative = F.pad(values.cumsum(-1, dtype=torch.float64), (1, 0))
    integral = cumulative.index_select(-1, indices) + values.index_select(
        -1, indices
    ) * (edges - indices)
    return (integral.diff(dim=-1) / (depth / size)).float()


def image_profile(images, direction=0, bins=None):
    if images.ndim != 4 or images.numel() == 0:
        raise ValueError("images must have non-empty shape [B,K,H,W].")
    if type(direction) is not int or direction not in (0, 1):
        raise ValueError("profile direction must be 0 (rows) or 1 (columns).")
    values = images.float().mean(dim=3 if direction == 0 else 2)
    return values if bins is None else rebin_profile(values, bins)


def validate_profile(spec, phases):
    if not isinstance(spec, dict) or set(spec) != {"axis", "points", "interpolation"}:
        raise ValueError("vf_profile requires axis, points and interpolation.")
    if spec["axis"] != "z" or spec["interpolation"] not in ("linear", "constant"):
        raise ValueError(
            "vf_profile requires z axis and linear or constant interpolation."
        )
    points = spec["points"]
    if not isinstance(points, (list, tuple)) or any(
        not isinstance(point, (list, tuple))
        or len(point) != 2
        or not isinstance(point[1], (list, tuple))
        for point in points
    ):
        raise ValueError("profile points must be [position, phase fractions] pairs.")
    try:
        knots = torch.tensor([p[0] for p in spec["points"]], dtype=torch.float64)
        values = torch.tensor([p[1] for p in spec["points"]], dtype=torch.float64)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("invalid profile points.") from exc
    if (
        knots.ndim != 1
        or len(knots) < 2
        or values.shape != (len(knots), phases)
        or not bool(torch.isfinite(knots).all() & torch.isfinite(values).all())
        or knots[0] != 0
        or knots[-1] != 1
        or not bool((knots.diff() > 0).all())
        or not bool(((values >= 0) & (values <= 1)).all())
        or not torch.allclose(
            values.sum(1), torch.ones(len(knots), dtype=values.dtype), atol=1e-6, rtol=0
        )
    ):
        raise ValueError(
            "profile points must span 0..1 with increasing knots and phase simplex values."
        )
    return knots, values


def sample_profile(spec, phases, depth, origin, spacing, extent, device=None):
    knots, values = validate_profile(spec, phases)
    edges = (origin + torch.arange(depth + 1, dtype=torch.float64) * spacing) / extent
    left, right = edges[:-1], edges[1:]
    result = torch.zeros(depth, phases, dtype=torch.float64)
    for index in range(len(knots) - 1):
        a = left.clamp(knots[index], knots[index + 1])
        b = right.clamp(knots[index], knots[index + 1])
        width = b - a
        result += width[:, None] * values[index]
        if spec["interpolation"] == "linear":
            slope = (values[index + 1] - values[index]) / (
                knots[index + 1] - knots[index]
            )
            result += (
                ((b - knots[index]).square() - (a - knots[index]).square())[:, None]
                * slope
                / 2
            )
    result += (right.clamp_max(0) - left.clamp_max(0))[:, None] * values[0]
    result += (right.clamp_min(1) - left.clamp_min(1))[:, None] * values[-1]
    return (
        (result / (spacing / extent))
        .T.unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def profile_field(profile, shape):
    if profile.ndim != 3 or profile.shape[-1] != shape[0]:
        raise ValueError("profile must match the volume depth.")
    return profile[..., None, None].expand(-1, -1, *shape)
