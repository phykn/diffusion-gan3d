import torch

# Each plane mirrors its second axis, so z is never reversed and height stays valid.
MIRROR_PLANES = {"zy": (0, 1), "zx": (0, 2), "yx": (1, 2)}
# Offset directions (along first, along mirrored axis) as multiples of the gap.
SLOPES = ((1, 1), (1, 2), (2, 1))


def phase_correlations(probs: torch.Tensor, reach: int) -> torch.Tensor:
    """Sums of probs[:, p](x) * probs[:, q](x + o) over x, indexed [b, p, q, *o].

    Zero padding keeps offsets up to reach along each axis free of wraparound;
    a negative offset sits at its index modulo the padded size.
    """
    depth, height, width = probs.shape[2:]
    sizes = [n + min(reach, n - 1) for n in (depth, height, width)]
    # Padding by hand: MPS pads five-dimensional tensors on a fallback that warns.
    padded = probs.new_zeros(*probs.shape[:2], *sizes)
    padded[..., :depth, :height, :width] = probs
    spectrum = torch.fft.rfftn(padded, dim=(-3, -2, -1))
    cross = spectrum.conj()[:, :, None] * spectrum[:, None]
    return torch.fft.irfftn(cross, s=sizes, dim=(-3, -2, -1))


def mirror_differences(
    correlations: torch.Tensor,
    shape: torch.Size,
    first: int,
    second: int,
    offsets: list[tuple[int, int]],
) -> torch.Tensor:
    """Per volume and offset (a, b), half the L1 distance between the phase-pair
    tables at (+a, +b) and (+a, -b) in one plane."""
    a, b = (torch.tensor(side, device=correlations.device) for side in zip(*offsets))
    sizes = correlations.shape[-3:]

    def at(along_first, along_second):
        index = [torch.zeros_like(a)] * 3
        index[first] = along_first % sizes[first]
        index[second] = along_second % sizes[second]
        return correlations[..., index[0], index[1], index[2]]

    # A table counts (p, q) and (q, p) alike, and the sum for (q, p) at o is
    # the sum for (p, q) at -o.
    plus = at(a, b) + at(-a, -b)
    minus = at(a, -b) + at(-a, b)
    voxels = (shape[first] - a) * (shape[second] - b) * shape[3 - first - second]
    return (plus - minus).abs().sum((1, 2)) * 0.25 / voxels


def compute_mirror_balance_loss(
    probs: torch.Tensor, max_gap: int, tolerance: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize per-volume differences between mirrored diagonal statistics.

    A mirror-symmetric structure has the same phase pairs along (+a, +b) as along
    (+a, -b). Axis-aligned statistics cannot see a shear such as f(y, x - z): it
    keeps xy and yz sections realistic and turns only xz sections into stripes.
    """
    zero = probs.sum(dtype=torch.float32) * 0.0
    diagnostics = {}
    losses = []
    with torch.autocast(device_type=probs.device.type, enabled=False):
        probs = probs.float()
        shape = probs.shape[2:]
        # One FFT gives the pair sums at every offset; on Apple MPS this is about
        # fifteen times faster than multiplying shifted windows offset by offset.
        correlations = phase_correlations(probs, 2 * max_gap)
        for name, (first, second) in MIRROR_PLANES.items():
            offsets = [
                (gap * slope_a, gap * slope_b)
                for gap in range(1, max_gap + 1)
                for slope_a, slope_b in SLOPES
                if gap * slope_a < shape[first] and gap * slope_b < shape[second]
            ]
            if not offsets:
                continue
            errors = mirror_differences(correlations, shape, first, second, offsets)
            losses.append((errors - tolerance).relu().mean())
            diagnostics[f"mirror_balance/{name}/difference"] = errors.mean().detach()
            diagnostics[f"mirror_balance/{name}/max"] = errors.amax(-1).mean().detach()
    return (torch.stack(losses).mean() if losses else zero), diagnostics
