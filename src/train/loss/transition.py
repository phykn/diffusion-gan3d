import torch

from src.plane import PLANE_DIRECTIONS, PLANES


def phase_pairs(images: torch.Tensor, dim: int, gap: int) -> torch.Tensor:
    """Joint phase probabilities per image row, without discretizing fractions."""
    length = images.shape[dim] - gap
    left = images.narrow(dim, 0, length).float()
    right = images.narrow(dim, gap, length).float()
    return torch.einsum("bcrw,bdrw->brcd", left, right) / left.shape[-1]


def height_intervals(height: torch.Tensor, dim: int, gap: int):
    rows = height[:, 0, :, 0].float()
    width = (rows[:, 1:] - rows[:, :-1]).mean(1, keepdim=True)
    centers = (rows[:, :-gap] + rows[:, gap:]) * 0.5 if dim == 2 else rows
    return (centers - width * 0.5).flatten(), (centers + width * 0.5).flatten()


def match_height(real, fake, real_height, fake_height, dim, gap):
    real = real.flatten(0, 1)
    fake = fake.flatten(0, 1)
    real_low, real_high = height_intervals(real_height, dim, gap)
    fake_low, fake_high = height_intervals(fake_height, dim, gap)
    errors, counts = [], []
    # Bound row matching memory independently of the number of sampled slices.
    for start in range(0, len(fake), 128):
        stop = start + 128
        overlap = (
            torch.minimum(fake_high[start:stop, None], real_high[None])
            - torch.maximum(fake_low[start:stop, None], real_low[None])
        ).clamp_min(0)
        weight = overlap.sum(1)
        target = torch.einsum("fr,rcd->fcd", overlap, real)
        target = target / weight.clamp_min(1e-8)[:, None, None]
        valid = weight > 1e-8
        error = 0.5 * (fake[start:stop] - target).abs().sum((1, 2))
        errors.append((error * valid).sum())
        counts.append(valid.sum())
    count = torch.stack(counts).sum()
    return torch.stack(errors).sum() / count.clamp_min(1), count


def compute_real_transition_loss(
    real: dict[int, torch.Tensor],
    fake: dict[int, torch.Tensor],
    max_gap: int,
    real_heights: dict[int, torch.Tensor] | None = None,
    fake_heights: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match observed in-plane transitions at equal LR-grid separations.

    Height-conditioned samples compare row statistics over overlapping normalized
    height intervals. No observations are inferred for a plane's normal axis.
    """
    zero = next(iter(fake.values())).sum(dtype=torch.float32) * 0.0
    groups: dict[tuple[str, int], list] = {}
    height_enabled = fake_heights is not None
    with torch.autocast(device_type=zero.device.type, enabled=False):
        for axis, generated in fake.items():
            if axis not in real or not len(generated):
                continue
            observed = real[axis].detach()
            if height_enabled and (
                axis not in real_heights
                or axis not in fake_heights
                or min(observed.shape[-2], generated.shape[-2]) < 2
            ):
                continue
            for dim, direction in enumerate(PLANE_DIRECTIONS[PLANES[axis]], 2):
                limit = min(max_gap + 1, observed.shape[dim], generated.shape[dim])
                for gap in range(1, limit):
                    target = phase_pairs(observed, dim, gap)
                    predicted = phase_pairs(generated, dim, gap)
                    if height_enabled:
                        error, count = match_height(
                            target,
                            predicted,
                            real_heights[axis],
                            fake_heights[axis],
                            dim,
                            gap,
                        )
                    else:
                        error = (
                            0.5
                            * (target.mean((0, 1)) - predicted.mean((0, 1))).abs().sum()
                        )
                        count = error.new_ones(())
                    groups.setdefault((direction, gap), []).append((error, count))
        losses, active = [], []
        diagnostics = {}
        for (direction, gap), values in groups.items():
            errors = torch.stack([error for error, _ in values])
            counts = torch.stack([count for _, count in values])
            valid = counts > 0
            loss = (errors * valid).sum() / valid.sum().clamp_min(1)
            losses.append(loss)
            active.append(valid.any())
            key = f"real_transition/{direction}/gap{gap}"
            diagnostics[key] = loss.detach()
            diagnostics[f"{key}/matches"] = counts.sum().detach()
        if not losses:
            return zero, diagnostics
        valid = torch.stack(active)
        total = (torch.stack(losses) * valid).sum() / valid.sum().clamp_min(1)
        return total, diagnostics
