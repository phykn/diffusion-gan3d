"""LR mirror_balance: mirrored diagonal pair tables must agree."""

import torch

from src.stats.pairs import mirror_differences, pair_correlations

# Each plane mirrors its second axis, so z is never reversed and height stays valid.
MIRROR_PLANES = {"zy": (0, 1), "zx": (0, 2), "yx": (1, 2)}
# Offset directions (along first, along mirrored axis) as multiples of the gap.
SLOPES = ((1, 1), (1, 2), (2, 1))


def mirror_balance_loss(
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
        correlations = pair_correlations(probs, 2 * max_gap)
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
