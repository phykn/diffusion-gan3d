import copy
import math
from numbers import Real

import torch
from torch import nn


def build_ema(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module.")

    average = copy.deepcopy(model)
    average.eval()
    for parameter in average.parameters():
        parameter.requires_grad_(False)
    return average


@torch.no_grad()
def update_ema(
    average: nn.Module,
    online: nn.Module,
    decay: float,
) -> None:
    if not isinstance(average, nn.Module) or not isinstance(online, nn.Module):
        raise TypeError("average and online must be torch modules.")
    if not isinstance(decay, Real) or isinstance(decay, bool):
        raise TypeError("decay must be a real scalar.")
    decay = float(decay)
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("decay must be finite and in [0, 1).")

    for target, source in zip(
        average.parameters(),
        online.parameters(),
        strict=True,
    ):
        target.lerp_(source, 1.0 - decay)

    for target, source in zip(
        average.buffers(),
        online.buffers(),
        strict=True,
    ):
        target.copy_(source)

    average.eval()
