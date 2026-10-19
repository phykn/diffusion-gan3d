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
    anchor_planes: int
    anchor_conflict_rate: float
    anchor_loss: float
    anchor_accuracy: float
    generator_connectivity: float
    critic_connectivity: float
    connectivity_r1: float
    anchor_ramp: float
    generator_connectivity_global: float = 0.0
    generator_connectivity_local: float = 0.0
    critic_connectivity_global: float = 0.0
    critic_connectivity_local: float = 0.0
    vf_loss: float = 0.0
    vf_active: bool = False
    anchor_input_active_fraction: float = 0.0
    vf_active_fraction: float = 0.0
    anchor_coarse_loss: float = 0.0
    anchor_pixel_loss: float = 0.0
    anchor_shared: bool = False
    connectivity_ramp: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def collect_metrics(
    volume_size: int,
    diagnostics: dict,
    prepared: StepPreparation,
    denoiser_update: DenoiserUpdate,
    critic_vals: list[float],
    r1: float,
    critic_connectivity: float,
    critic_connectivity_global: float,
    critic_connectivity_local: float,
    connectivity_r1: float,
) -> Metrics:
    selection = prepared.selection
    anchor = None if selection is None else selection.condition
    presence = prepared.presence
    metrics = materialize_metrics(
        Metrics(
            generator=denoiser_update.adversarial,
            generator_total=denoiser_update.total,
            critic=sum(critic_vals),
            r1=r1,
            transition=prepared.transition,
            volume_size=volume_size,
            domain=prepared.domain,
            critic_axes=tuple(critic_vals),
            anchor_planes=0 if anchor is None else anchor.planes,
            anchor_conflict_rate=0.0 if anchor is None else anchor.conflict_rate,
            anchor_loss=denoiser_update.anchor,
            anchor_accuracy=denoiser_update.anchor_accuracy,
            generator_connectivity=denoiser_update.connectivity,
            generator_connectivity_global=denoiser_update.connectivity_global,
            generator_connectivity_local=denoiser_update.connectivity_local,
            critic_connectivity=critic_connectivity,
            critic_connectivity_global=critic_connectivity_global,
            critic_connectivity_local=critic_connectivity_local,
            connectivity_r1=connectivity_r1,
            anchor_ramp=prepared.anchor_ramp,
            connectivity_ramp=prepared.connectivity_ramp,
            vf_loss=denoiser_update.vf,
            vf_active=presence.vf.any(),
            anchor_input_active_fraction=presence.anchor.to(torch.float32).mean(),
            vf_active_fraction=presence.vf.to(torch.float32).mean(),
            anchor_coarse_loss=denoiser_update.anchor_coarse,
            anchor_pixel_loss=denoiser_update.anchor_pixel,
            anchor_shared=(
                prepared.selection is not None
                and prepared.selection.source == "shared"
                and presence.anchor.any()
            ),
            diagnostics=dict(diagnostics),
        )
    )
    if metrics.diagnostics.get("anchor/boundary_pairs", 0) == 0:
        for key in ("anchor/neighbor_agreement", "anchor/neighbor_excess_jump"):
            if key in metrics.diagnostics:
                metrics.diagnostics[key] = None
    return metrics


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
        "loss/generator_connectivity": metrics.generator_connectivity,
        "loss/generator_connectivity_global": metrics.generator_connectivity_global,
        "loss/generator_connectivity_local": metrics.generator_connectivity_local,
        "loss/critic_connectivity": metrics.critic_connectivity,
        "loss/critic_connectivity_global": metrics.critic_connectivity_global,
        "loss/critic_connectivity_local": metrics.critic_connectivity_local,
        "loss/connectivity_r1": metrics.connectivity_r1,
        "loss/anchor": metrics.anchor_loss,
        "loss/anchor_coarse": metrics.anchor_coarse_loss,
        "loss/anchor_pixel": metrics.anchor_pixel_loss,
        "loss/vf": metrics.vf_loss,
        "conditioning/vf_active": metrics.vf_active,
        "conditioning/anchor_planes": metrics.anchor_planes,
        "conditioning/anchor_accuracy": metrics.anchor_accuracy,
        "conditioning/anchor_conflict_rate": metrics.anchor_conflict_rate,
        "conditioning/anchor_shared": metrics.anchor_shared,
        "conditioning/anchor_fraction": metrics.anchor_input_active_fraction,
        "conditioning/vf_fraction": metrics.vf_active_fraction,
        "conditioning/anchor_ramp": metrics.anchor_ramp,
        "conditioning/connectivity_ramp": metrics.connectivity_ramp,
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
