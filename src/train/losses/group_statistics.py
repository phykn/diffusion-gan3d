"""Pair tables along the normals of the planes one critic judges must agree."""

from itertools import combinations

import torch

from src.data.crops import resize_phases
from src.stats.pairs import volume_pairs


def group_statistics_loss(
    probs: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
    max_gap: int,
    tolerance: float,
    preserve_height: bool = False,
    coarse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize per-volume phase-pair differences between plane normals beyond
    tolerance (plus, for SR, the difference the coarse input already has)."""
    zero = probs.sum(dtype=torch.float32) * 0.0
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
                                volume_pairs(probs, axis, gap, preserve_height),
                                None
                                if reference is None
                                else volume_pairs(
                                    reference, axis, gap, preserve_height
                                ),
                            )
                    a, ref_a = cache[first, gap]
                    b, ref_b = cache[second, gap]
                    error = (a - b).abs().sum((-2, -1)) * 0.5
                    allowance = tolerance
                    if reference is not None:
                        allowance = (
                            allowance + (ref_a - ref_b).abs().sum((-2, -1)) * 0.5
                        )
                    # ReLU has zero gradient at the allowance boundary as well.
                    penalty = (error - allowance).relu()
                    errors.append(error.mean())
                    penalties.append(penalty.mean())
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
