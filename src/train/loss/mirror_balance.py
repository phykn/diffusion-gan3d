import torch

# Each plane mirrors its second axis, so z is never reversed and height stays valid.
MIRROR_PLANES = {"zy": (0, 1), "zx": (0, 2), "yx": (1, 2)}
# Offset directions (along first, along mirrored axis) as multiples of the gap.
SLOPES = ((1, 1), (1, 2), (2, 1))


def diagonal_phase_pairs(probs: torch.Tensor, first: int, second: int, a: int, b: int):
    """Phase-pair probabilities at offsets (+a, +b) and (+a, -b) in one plane."""
    first, second = first + 2, second + 2
    length_a = probs.shape[first] - a
    length_b = probs.shape[second] - b

    def window(start_a, start_b):
        view = probs.narrow(first, start_a, length_a)
        return view.narrow(second, start_b, length_b)

    pairs = []
    for left, right in (
        (window(0, 0), window(a, b)),
        (window(0, b), window(a, 0)),
    ):
        rows = [
            (left[:, phase : phase + 1] * right).mean((-3, -2, -1))
            for phase in range(probs.shape[1])
        ]
        table = torch.stack(rows, dim=-2)
        pairs.append((table + table.transpose(-1, -2)) * 0.5)
    return pairs


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
        for name, (first, second) in MIRROR_PLANES.items():
            errors = []
            for gap in range(1, max_gap + 1):
                for slope_a, slope_b in SLOPES:
                    a, b = gap * slope_a, gap * slope_b
                    if a >= probs.shape[first + 2] or b >= probs.shape[second + 2]:
                        continue
                    plus, minus = diagonal_phase_pairs(probs, first, second, a, b)
                    errors.append((plus - minus).abs().sum((-2, -1)) * 0.5)
            if not errors:
                continue
            errors = torch.stack(errors, dim=-1)
            losses.append((errors - tolerance).relu().mean())
            diagnostics[f"mirror_balance/{name}/difference"] = errors.mean().detach()
            diagnostics[f"mirror_balance/{name}/max"] = errors.amax(-1).mean().detach()
    return (torch.stack(losses).mean() if losses else zero), diagnostics
