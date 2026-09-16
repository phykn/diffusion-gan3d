import math
from dataclasses import fields, is_dataclass

import torch


def materialize_metrics(value):
    """Transfer scalar diagnostics together after the optimizer work is complete."""
    tensors = []

    def collect(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item.detach())
        elif is_dataclass(item):
            for field in fields(item):
                collect(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    if not tensors:
        return value
    device = tensors[0].device
    values = iter(
        torch.stack([t.to(device=device, dtype=torch.float64) for t in tensors])
        .cpu()
        .tolist()
    )

    def rebuild(item):
        if isinstance(item, torch.Tensor):
            result = next(values)
            if item.dtype == torch.bool:
                return bool(result)
            return float(result) if item.dtype.is_floating_point else int(result)
        if is_dataclass(item):
            return type(item)(
                **{
                    field.name: rebuild(getattr(item, field.name))
                    for field in fields(item)
                }
            )
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return type(item)(rebuild(child) for child in item)
        return item

    return rebuild(value)


def check_loss(loss: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite {name} loss; optimizer was not stepped.")


def step_optimizer(optimizer, scaler, diagnostics: dict, name: str) -> bool:
    if scaler is not None:
        scaler.unscale_(optimizer)
    grads = [
        p.grad.detach().float()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    if not grads:
        raise RuntimeError(f"{name} has no gradients.")
    norm = float(torch.stack([g.norm(dtype=torch.float64) for g in grads]).norm())
    finite = math.isfinite(norm)
    diagnostics[f"gradient/{name}"] = norm if finite else None
    diagnostics[f"skipped/{name}"] = int(not finite)
    if not finite and (scaler is None or not scaler.is_enabled()):
        raise FloatingPointError(
            f"non-finite {name} gradient; optimizer was not stepped."
        )
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
    return finite
