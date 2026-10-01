from itertools import combinations

import torch

from src.prepare.resize import resize_phases


def phase_pairs(probs: torch.Tensor, axis: int, gap: int, preserve_height: bool):
    dim = axis + 2
    length = probs.shape[dim] - gap
    left = probs.narrow(dim, 0, length)
    right = probs.narrow(dim, gap, length)
    dims = (-2, -1) if preserve_height else (-3, -2, -1)
    rows = []
    for phase in range(probs.shape[1]):
        row = (left[:, phase : phase + 1] * right).mean(dims)
        rows.append(row.movedim(1, -1) if preserve_height else row)
    # Slice views avoid retaining a flattened volume copy for each axis/gap.
    pairs = torch.stack(rows, dim=-2)
    # Reversing an axis changes pair order, not its undirected structure.
    return (pairs + pairs.transpose(-1, -2)) * 0.5


def compute_group_statistics_loss(
    probs: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
    max_gap: int,
    tolerance: float,
    preserve_height: bool = False,
    active: torch.Tensor | None = None,
    coarse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize excess per-volume phase-pair differences between plane normals."""
    zero = probs.sum(dtype=torch.float32) * 0.0
    if active is None:
        active = probs.new_ones(probs.shape[0])
    active = active.to(device=probs.device, dtype=torch.float32)
    count = active.sum().clamp_min(1)
    diagnostics = {}
    losses = []
    with torch.autocast(device_type=probs.device.type, enabled=False):
        probs = probs.float()
        reference = (
            None
            if coarse is None
            else resize_phases(coarse.detach().float(), tuple(probs.shape[-3:]))
        )
        cache = {}
        for name, axes in groups.items():
            axes = tuple(axis for axis in axes if not preserve_height or axis != 0)
            comparisons = list(combinations(axes, 2))
            group_losses = []
            for gap in range(1, max_gap + 1):
                errors, penalties = [], []
                for first, second in comparisons:
                    if gap >= min(probs.shape[first + 2], probs.shape[second + 2]):
                        continue
                    for axis in (first, second):
                        key = (axis, gap)
                        if key not in cache:
                            cache[key] = (
                                phase_pairs(probs, axis, gap, preserve_height),
                                None
                                if reference is None
                                else phase_pairs(reference, axis, gap, preserve_height),
                            )
                    a, ref_a = cache[first, gap]
                    b, ref_b = cache[second, gap]
                    error = (a - b).abs().sum((-2, -1)) * 0.5
                    allowance = tolerance
                    if reference is not None:
                        allowance = (
                            allowance + (ref_a - ref_b).abs().sum((-2, -1)) * 0.5
                        )
                    penalty = (error - allowance).clamp_min(0)
                    errors.append(
                        (error.reshape(len(probs), -1).mean(1) * active).sum() / count
                    )
                    penalties.append(
                        (penalty.reshape(len(probs), -1).mean(1) * active).sum() / count
                    )
                if penalties:
                    value = torch.stack(penalties).mean()
                    group_losses.append(value)
                    prefix = f"group_statistics/{name}/gap{gap}"
                    diagnostics[f"{prefix}/difference"] = (
                        torch.stack(errors).mean().detach()
                    )
                    diagnostics[f"{prefix}/penalty"] = value.detach()
            if group_losses:
                losses.append(torch.stack(group_losses).mean())
    return (torch.stack(losses).mean() if losses else zero), diagnostics
