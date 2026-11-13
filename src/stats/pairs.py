import torch


def section_pairs(images, dim, gap):
    """Pair tables per row of [B, K, H, W] sections along dim (2 or 3): [B, R, K, K]."""
    length = images.shape[dim] - gap
    left = images.narrow(dim, 0, length).float()
    right = images.narrow(dim, gap, length).float()
    return torch.einsum("bcrw,bdrw->brcd", left, right) / left.shape[-1]


def volume_pairs(probs, axis, gap, preserve_height=False):
    """Undirected pair tables of [B, K, D, H, W] volumes along a volume axis.

    [B, K, K], or [B, D, K, K] per z when preserve_height keeps the z layers.
    """
    dim = axis + 2
    length = probs.shape[dim] - gap
    left = probs.narrow(dim, 0, length)
    right = probs.narrow(dim, gap, length)
    dims = (-2, -1) if preserve_height else (-3, -2, -1)
    # Slice views avoid retaining a flattened volume copy for each axis/gap.
    rows = [(phase * right).mean(dims) for phase in left.split(1, dim=1)]
    pairs = torch.stack(rows, dim=1)
    if preserve_height:
        pairs = pairs.movedim(-1, 1)
    return (pairs + pairs.transpose(-1, -2)) * 0.5


def pair_correlations(probs, reach):
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


def mirror_differences(correlations, shape, first, second, offsets):
    """Per volume and offset (a, b), half the L1 distance between the pair tables
    at (+a, +b) and (+a, -b) in the plane of volume axes first and second."""
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
