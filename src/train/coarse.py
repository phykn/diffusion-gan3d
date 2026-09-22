import torch
import torch.nn.functional as F

from src.prepare.resize import phase_channels


def corrupt_coarse(low, probability, strength):
    if probability == 0 or strength == 0:
        return low, low.new_zeros(len(low))
    shape = (low.shape[0], 1, *(max(1, n // 4) for n in low.shape[2:]))
    active = torch.rand((len(low), 1, 1, 1, 1), device=low.device) < probability
    level = torch.rand((len(low), 1, 1, 1, 1), device=low.device) * strength * active
    mask = (
        F.interpolate(
            (torch.rand(shape, device=low.device) < level).float(),
            size=low.shape[2:],
            mode="nearest",
        ).bool()
        & active
    )
    labels = torch.randint(low.shape[1], (shape[0], *shape[2:]), device=low.device)
    replacement = phase_channels(labels, low.shape[1])
    replacement = F.interpolate(replacement, size=low.shape[2:], mode="nearest")
    corrupted = torch.where(mask, replacement, low)
    # Matching replacements do not contribute to the realized corruption level.
    level = (corrupted - low).abs().mean(dim=(2, 3, 4)).sum(1) * 0.5
    return corrupted, level
