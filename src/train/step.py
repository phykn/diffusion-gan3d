import torch


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
    norm = torch.stack([g.norm(dtype=torch.float64) for g in grads]).norm()
    finite = bool(torch.isfinite(norm))
    diagnostics[f"gradient/{name}"] = float(norm) if finite else None
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
