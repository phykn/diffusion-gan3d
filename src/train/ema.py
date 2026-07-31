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

    average_params = dict(average.named_parameters())
    online_params = dict(online.named_parameters())
    _require_matching_state("parameters", average_params, online_params)
    for name, target in average_params.items():
        source = online_params[name].detach()
        _require_compatible_tensor(name, target, source)
        target.lerp_(source, 1.0 - decay)

    average_buffers = dict(average.named_buffers())
    online_buffers = dict(online.named_buffers())
    _require_matching_state("buffers", average_buffers, online_buffers)
    for name, target in average_buffers.items():
        source = online_buffers[name].detach()
        _require_compatible_tensor(name, target, source)
        target.copy_(source)

    average.eval()


def _require_matching_state(
    label: str,
    average: dict[str, torch.Tensor],
    online: dict[str, torch.Tensor],
) -> None:
    if average.keys() != online.keys():
        raise ValueError(f"EMA {label} must match the online model exactly.")


def _require_compatible_tensor(
    name: str,
    target: torch.Tensor,
    source: torch.Tensor,
) -> None:
    if (
        target.shape != source.shape
        or target.dtype != source.dtype
        or target.device != source.device
    ):
        raise ValueError(
            f"EMA tensor {name} must match the online tensor shape, dtype, and device."
        )
