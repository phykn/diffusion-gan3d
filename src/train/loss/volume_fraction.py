import torch


def compute_vf_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    present = present.to(probs.device)
    predicted = probs.to(torch.float32).mean(dim=(2, 3, 4))
    target = target.to(torch.float32)
    target_log = torch.where(
        target > 0.0,
        target.log(),
        torch.zeros_like(target),
    )
    predicted_log = predicted.clamp_min(1e-6).log()
    per_sample = (target * (target_log - predicted_log)).sum(dim=1)
    return (per_sample * present).sum() / present.sum().clamp_min(1)
