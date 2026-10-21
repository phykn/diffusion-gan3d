import math

import torch

from src.device import wide_dtype


def input_gradient_norms(loss: torch.Tensor, inputs: tuple[torch.Tensor, ...]):
    active = tuple(value for value in inputs if value.requires_grad)
    grads = iter(
        torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True)
        if active and loss.requires_grad
        else (None,) * len(active)
    )
    norms = []
    for value in inputs:
        grad = next(grads) if value.requires_grad else None
        norms.append(
            loss.new_zeros(())
            if grad is None
            else grad.detach().float().flatten(1).norm(dim=1).mean()
        )
    return tuple(norms)


def validate_loss(loss: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite {name} loss; optimizer was not stepped.")
    return loss


def step_optimizer(optimizer, scaler, diagnostics: dict, name: str) -> bool:
    scaler.unscale_(optimizer)
    grads = [
        p.grad.detach().float()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    if not grads:
        raise RuntimeError(f"{name} has no gradients.")
    dtype = wide_dtype(grads[0].device)
    norm = float(
        torch.stack([torch.linalg.vector_norm(g, dtype=dtype) for g in grads]).norm()
    )
    finite = math.isfinite(norm)
    diagnostics[f"gradient/{name}"] = norm if finite else None
    diagnostics[f"skipped/{name}"] = int(not finite)
    if not finite and not scaler.is_enabled():
        raise FloatingPointError(
            f"non-finite {name} gradient; optimizer was not stepped."
        )
    scaler.step(optimizer)
    return finite
