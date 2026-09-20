from dataclasses import dataclass, field, fields, is_dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

from src.plane import PLANES


@dataclass(frozen=True)
class Metrics:
    generator: float
    generator_total: float
    critic: float
    r1: float
    transition: int
    volume_size: int
    domain: int
    critic_axes: tuple[float, float, float]
    anchor_planes: int
    anchor_conflict_rate: float
    anchor_loss: float
    anchor_accuracy: float
    generator_connectivity: float
    critic_connectivity: float
    connectivity_r1: float
    anchor_ramp: float
    generator_global: float = 0.0
    generator_local: float = 0.0
    critic_global: float = 0.0
    critic_local: float = 0.0
    vf_loss: float = 0.0
    vf_active: bool = False
    anchor_input_active_fraction: float = 0.0
    vf_active_fraction: float = 0.0
    normal_transition_loss: float = 0.0
    anchor_coarse_loss: float = 0.0
    anchor_pixel_loss: float = 0.0
    anchor_shared: bool = False
    connectivity_ramp: float = 0.0
    anchor_neighbor_agreement: float | None = None
    anchor_neighbor_excess_jump: float | None = None
    diagnostics: dict = field(default_factory=dict)


def materialize_metrics(value):
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


def write_metrics(writer: SummaryWriter, step: int, metrics: Metrics) -> None:
    scalars = {
        "loss/generator": metrics.generator,
        "loss/generator_total": metrics.generator_total,
        "loss/critic": metrics.critic,
        "loss/r1": metrics.r1,
        "loss/generator_connectivity": metrics.generator_connectivity,
        "loss/critic_connectivity": metrics.critic_connectivity,
        "loss/connectivity_r1": metrics.connectivity_r1,
        "loss/normal_transition": metrics.normal_transition_loss,
        "loss/anchor": metrics.anchor_loss,
        "loss/vf": metrics.vf_loss,
        "conditioning/anchor_fraction": metrics.anchor_input_active_fraction,
        "conditioning/vf_fraction": metrics.vf_active_fraction,
        "conditioning/anchor_ramp": metrics.anchor_ramp,
        "conditioning/connectivity_ramp": metrics.connectivity_ramp,
        "sampling/transition": metrics.transition,
        f"timestep/{metrics.transition}/generator": metrics.generator,
        f"timestep/{metrics.transition}/critic": metrics.critic,
        **{
            f"critic_plane/{PLANES[axis]}": value
            for axis, value in enumerate(metrics.critic_axes)
        },
        **metrics.diagnostics,
    }

    if metrics.anchor_planes:
        scalars["conditioning/anchor_planes"] = metrics.anchor_planes
        scalars["conditioning/anchor_accuracy"] = metrics.anchor_accuracy

    for tag, value in scalars.items():
        if value is not None:
            writer.add_scalar(tag, value, step)
