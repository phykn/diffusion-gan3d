import math

import torch
import torch.nn.functional as F

from .. import AXES
from ..anchor import AnchorCondition


def build_pool(
    batches: dict[int, torch.Tensor],
    num_phases: int,
) -> torch.Tensor:
    if not batches or not set(batches).issubset(AXES):
        raise ValueError("batches must contain at least one valid axis.")
    if any(
        not isinstance(images, torch.Tensor) or images.ndim != 3
        for images in batches.values()
    ):
        raise ValueError("training crops must have shape [B, H, W].")
    batch_sizes = {images.shape[0] for images in batches.values()}
    if len(batch_sizes) != 1 or not batch_sizes or next(iter(batch_sizes)) < 1:
        raise ValueError(
            "each available axis must provide the same non-empty crop batch."
        )

    values = []
    for images in batches.values():
        if images.numel() == 0:
            raise ValueError("training crops must not be empty.")
        labels = images.to(torch.long)
        lower, upper = torch.aminmax(labels)
        if int(lower) < 0 or int(upper) >= num_phases:
            raise ValueError("training images contain a phase outside num_phases.")
        batch = labels.shape[0]
        offsets = torch.arange(
            batch,
            device=labels.device,
            dtype=labels.dtype,
        ).mul_(num_phases)
        encoded = labels.flatten(1).add(offsets[:, None])
        counts = torch.bincount(
            encoded.flatten(),
            minlength=batch * num_phases,
        ).reshape(batch, num_phases)
        fractions = counts.to(torch.float32).div_(labels.shape[-2] * labels.shape[-1])
        values.append(fractions)
    return torch.cat(values, dim=0)


def validate_target(
    target: torch.Tensor,
    batch_size: int,
    num_phases: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(target, torch.Tensor) or not target.is_floating_point():
        raise TypeError("target VF must be a floating-point tensor.")
    expected = (batch_size, num_phases)
    if target.shape != expected:
        raise ValueError(f"target VF must have shape {expected}.")
    target = target.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(target).all()) or bool((target < 0.0).any()):
        raise ValueError("target VF values must be finite and non-negative.")
    sums = target.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=0.0):
        raise ValueError("target VF rows must sum to one.")
    return target


def sample_target(
    pool: torch.Tensor,
    anchor: AnchorCondition | None,
    batch_size: int,
    num_phases: int,
    device: torch.device,
    max_samples: int,
) -> tuple[torch.Tensor, float]:
    if (
        not isinstance(pool, torch.Tensor)
        or not pool.is_floating_point()
        or pool.ndim != 2
        or pool.shape[0] < 1
        or pool.shape[1] != num_phases
    ):
        raise ValueError("VF pool must have shape [N, num_phases].")
    pool = pool.to(device=device, dtype=torch.float32)
    max_samples = min(max_samples, pool.shape[0])
    sample_counts = (
        torch.ones(
            batch_size,
            device=device,
            dtype=torch.long,
        )
        if max_samples == 1
        else torch.randint(
            1,
            max_samples + 1,
            (batch_size,),
            device=device,
        )
    )
    indices = torch.randint(
        pool.shape[0],
        (batch_size, max_samples),
        device=device,
    )
    active = torch.arange(max_samples, device=device)[None] < sample_counts[:, None]
    target = (pool[indices] * active[..., None]).sum(dim=1).div_(sample_counts[:, None])
    resampled = 0
    if anchor is not None:
        if anchor.target.shape[0] != batch_size:
            raise ValueError("anchor and generated volume batches must match.")
        required = anchor_minimum(anchor, batch_size, num_phases)
        for batch, minimum in enumerate(required):
            valid = (pool + 1e-7 >= minimum).all(dim=1).nonzero().flatten()
            if not len(valid):
                raise ValueError(
                    "anchor phase minima are incompatible with every empirical "
                    "VF target."
                )
            if not bool((target[batch] + 1e-7 >= minimum).all()):
                choices = torch.randint(
                    len(valid),
                    (int(sample_counts[batch]),),
                    device=device,
                )
                target[batch] = pool[valid.index_select(0, choices)].mean(dim=0)
                resampled += 1
    target = validate_target(target, batch_size, num_phases, device)
    return target, resampled / batch_size


def anchor_is_compatible(
    anchor: AnchorCondition,
    pool: torch.Tensor,
    batch_size: int,
    num_phases: int,
) -> bool:
    """Whether every anchor sample fits at least one target-domain VF."""
    required = anchor_minimum(anchor, batch_size, num_phases)
    pool = pool.to(device=required.device, dtype=torch.float32)
    compatible = (pool.unsqueeze(0) + 1e-7 >= required.unsqueeze(1)).all(dim=2)
    return bool(compatible.any(dim=1).all())


def anchor_minimum(
    anchor: AnchorCondition,
    batch_size: int,
    num_phases: int,
) -> torch.Tensor:
    if anchor.target.shape[0] != batch_size:
        raise ValueError("anchor and generated volume batches must match.")
    voxel_count = math.prod(anchor.target.shape[1:])
    required = []
    for labels, mask in zip(
        anchor.target,
        anchor.mask[:, 0].to(torch.bool),
        strict=True,
    ):
        counts = torch.bincount(
            labels[mask].to(torch.long),
            minlength=num_phases,
        ).to(torch.float32)
        required.append(counts / voxel_count)
    return torch.stack(required)


def summarize(
    probs: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
]:
    if present.dtype != torch.bool or present.shape != (probs.shape[0],):
        raise ValueError("VF presence must be a boolean tensor with shape [B].")
    values = probs.detach().to(torch.float32)
    target_values = target.detach().to(torch.float32).mean(dim=0)
    if bool(present.any()):
        values = values[present]
        selected_target = target[present].to(torch.float32)
        soft_values = values.mean(dim=(0, 2, 3, 4))
        phases = values.argmax(dim=1)
        hard_per_sample = (
            F.one_hot(phases, num_classes=values.shape[1])
            .to(torch.float32)
            .mean(dim=(1, 2, 3))
        )
        hard_values = hard_per_sample.mean(dim=0)
        hard_mae = (hard_per_sample - selected_target).abs().mean()
    else:
        soft_values = torch.zeros_like(target_values)
        hard_values = torch.zeros_like(target_values)
        hard_mae = torch.zeros((), device=values.device)
    return (
        tuple(float(value) for value in target_values),
        tuple(float(value) for value in soft_values),
        tuple(float(value) for value in hard_values),
        float(hard_mae),
    )
