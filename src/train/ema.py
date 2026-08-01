import copy

import torch
from torch import nn


def build_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(
    ema: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    for ema_parameter, model_parameter in zip(
        ema.parameters(),
        model.parameters(),
        strict=True,
    ):
        ema_parameter.lerp_(model_parameter, 1.0 - decay)

    for ema_buffer, model_buffer in zip(
        ema.buffers(),
        model.buffers(),
        strict=True,
    ):
        ema_buffer.copy_(model_buffer)

    ema.eval()
