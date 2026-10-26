from dataclasses import dataclass, field, fields, is_dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

from src.device import wide_dtype
from src.plane import PLANES
from src.train.batch import DenoiserUpdate, StepPreparation


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
    vf_loss: float = 0.0
    vf_active: bool = False
    vf_active_fraction: float = 0.0
    real_transition_ramp: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def collect_metrics(
    volume_size: int,
    diagnostics: dict,
    prepared: StepPreparation,
    denoiser_update: DenoiserUpdate,
    critic_vals: list[float],
    r1: float,
) -> Metrics:
    return materialize_metrics(
        Metrics(
            generator=denoiser_update.adversarial,
            generator_total=denoiser_update.total,
            critic=sum(critic_vals),
            r1=r1,
            transition=prepared.transition,
            volume_size=volume_size,
            domain=prepared.domain,
            critic_axes=tuple(critic_vals),
            real_transition_ramp=prepared.real_transition_ramp,
            vf_loss=denoiser_update.vf,
            vf_active=prepared.vf_present.any(),
            vf_active_fraction=prepared.vf_present.to(torch.float32).mean(),
            diagnostics=dict(diagnostics),
        )
    )


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
    dtype = wide_dtype(device)
    values = iter(
        torch.stack([t.to(device=device, dtype=dtype) for t in tensors]).cpu().tolist()
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
        "loss/vf": metrics.vf_loss,
        "conditioning/vf_active": metrics.vf_active,
        "conditioning/vf_fraction": metrics.vf_active_fraction,
        "loss/real_transition_ramp": metrics.real_transition_ramp,
        "sampling/transition": metrics.transition,
        "sampling/volume_size": metrics.volume_size,
        "sampling/domain": metrics.domain,
        f"timestep/{metrics.transition}/generator": metrics.generator,
        f"timestep/{metrics.transition}/critic": metrics.critic,
        **{
            f"critic_plane/{PLANES[axis]}": value
            for axis, value in enumerate(metrics.critic_axes)
        },
        **metrics.diagnostics,
    }

    for tag, value in scalars.items():
        # Unavailable measurements remain visible as gaps at their recorded steps.
        writer.add_scalar(tag, float("nan") if value is None else value, step)
