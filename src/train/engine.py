import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .. import AXES
from ..anchor import AnchorCondition, PlaneAnchor, build_anchors
from ..dataset import BatchStream
from ..diffusion import Diffusion
from ..model.denoiser import Denoiser3D
from ..model.domain import NULL_DOMAIN
from .anchor_loss import soft_anchor_loss
from .augment import CriticAugment
from .connect import (
    TEACHER_MIN_ENTRIES,
    Connectivity,
    TripletBatch,
    normal_transition_loss,
)
from .ema import update_ema
from .loss import (
    get_critic_loss,
    get_critic_r1,
    get_generator_loss,
)
from .relation import RelationBank


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
    connectivity_triplets: int
    connectivity_replay: int
    anchor_teacher: bool
    teacher_volumes: int
    teacher_mebibytes: float
    generator_global: float = 0.0
    generator_local: float = 0.0
    critic_global: float = 0.0
    critic_local: float = 0.0
    vf_loss: float = 0.0
    vf_active: bool = False
    target_vfs: tuple[float, ...] = ()
    target_vf_stds: tuple[float, ...] = ()
    soft_vfs: tuple[float, ...] = ()
    hard_vfs: tuple[float, ...] = ()
    hard_vf_mae: float = 0.0
    vf_target_resample_rate: float = 0.0
    anchor_input_active_fraction: float = 0.0
    vf_active_fraction: float = 0.0
    condition_state_fractions: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    normal_transition_loss: float = 0.0
    anchor_coarse_loss: float = 0.0
    anchor_pixel_loss: float = 0.0
    relation_loss: float = 0.0
    relation_weighted_loss: float = 0.0
    relation_time_weight: float = 0.0
    relation_phase_loss: float = 0.0
    relation_support_loss: float = 0.0
    relation_minus_loss: float = 0.0
    relation_plus_loss: float = 0.0
    relation_queries: int = 0
    relation_matches: int = 0
    relation_domain_matches: int = 0
    relation_shared_matches: int = 0
    relation_ood_rejections: int = 0
    relation_missing_references: int = 0
    relation_distance_weights: tuple[float, ...] = ()
    relation_bank_entries: int = 0
    relation_ready_buckets: int = 0
    relation_prior_ready: bool = False
    anchor_shared: bool = False


@dataclass(frozen=True)
class DenoiserUpdate:
    adversarial: float
    total: float
    global_loss: float
    local_loss: float
    connectivity: float
    normal_transition: float
    anchor: float
    anchor_coarse: float
    anchor_pixel: float
    anchor_accuracy: float
    relation: float
    relation_weighted: float
    relation_time_weight: float
    relation_phase: float
    relation_support: float
    relation_minus: float
    relation_plus: float
    relation_queries: int
    relation_matches: int
    relation_domain_matches: int
    relation_shared_matches: int
    relation_ood_rejections: int
    relation_missing_references: int
    relation_distance_weights: tuple[float, ...]
    vf: float


@dataclass(frozen=True)
class DenoiserBatch:
    transition: int
    model_domain: int
    critic_domains: dict[int, int]
    fake: dict[int, tuple[torch.Tensor, torch.Tensor]]
    connectivity_real: TripletBatch
    connectivity_fake: TripletBatch
    logits: torch.Tensor
    clean_probs: torch.Tensor
    anchor: AnchorCondition | None
    anchor_present: torch.Tensor
    anchor_ramp: float
    target_vf: torch.Tensor
    vf_present: torch.Tensor


@dataclass(frozen=True)
class StepPreparation:
    transition: int
    domain: int
    model_domain: int
    critic_domains: dict[int, int]
    real: dict[int, torch.Tensor]
    selection: "AnchorSelection | None"
    anchor: AnchorCondition | None
    target_vf: torch.Tensor
    presence: "ConditionPresence"
    model_conditions: dict[str, torch.Tensor]
    anchor_ramp: float
    vf_target_resample_rate: float


@dataclass(frozen=True)
class AnchorSelection:
    condition: AnchorCondition
    source: Literal["real", "shared", "teacher"]
    seeds: tuple[PlaneAnchor, ...] = ()
    target_vf: torch.Tensor | None = None


@dataclass(frozen=True)
class ConditionPresence:
    anchor: torch.Tensor
    vf: torch.Tensor

    def __post_init__(self) -> None:
        if self.anchor.dtype != torch.bool or self.vf.dtype != torch.bool:
            raise TypeError("condition presence masks must be boolean tensors.")
        if self.anchor.ndim != 1 or self.anchor.shape != self.vf.shape:
            raise ValueError("condition presence masks must have matching shape [B].")
        if self.anchor.device != self.vf.device:
            raise ValueError("condition presence masks must be on the same device.")

    def fractions(self) -> tuple[float, float, float, float]:
        anchor = self.anchor
        vf = self.vf
        states = (
            anchor & vf,
            anchor & ~vf,
            ~anchor & vf,
            ~anchor & ~vf,
        )
        return tuple(float(state.to(torch.float32).mean()) for state in states)


@dataclass(frozen=True)
class TrainerComponents:
    denoiser: Denoiser3D
    ema_denoiser: Denoiser3D
    critics: nn.ModuleDict
    connectivity_critic: nn.Module
    streams: dict[int, dict[int, BatchStream]]
    diffusion: Diffusion
    denoiser_optim: torch.optim.Optimizer
    critic_optims: dict[str, torch.optim.Optimizer]
    connectivity_optim: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    device: torch.device
    critic_augment: CriticAugment | None = None

    def __post_init__(self) -> None:
        if set(self.streams) != set(range(len(self.streams))):
            raise ValueError("stream domain IDs must be contiguous and start at zero.")
        if any(not streams for streams in self.streams.values()):
            raise ValueError("each domain must contain at least one axis stream.")
        if any(not set(streams).issubset(AXES) for streams in self.streams.values()):
            raise ValueError("domain streams may contain only axes 0, 1, and 2.")
        available = {axis for streams in self.streams.values() for axis in streams}
        expected = {str(axis) for axis in available}
        if set(self.critics) != expected or set(self.critic_optims) != expected:
            raise ValueError("critics and optimizers must match the available axes.")
        if self.critic_augment is not None and not isinstance(
            self.critic_augment,
            CriticAugment,
        ):
            raise TypeError("critic_augment must be a CriticAugment or None.")


@dataclass(frozen=True)
class TrainerSettings:
    volume_batch_size: int
    num_phases: int
    patch_size: int
    slice_pairs_per_axis: int
    ema_decay: float
    r1_gamma: float
    r1_interval: int
    critic_local_weight: float
    anchor_training_probability: float
    anchor_start_step: int
    anchor_ramp_steps: int
    anchor_multi_probability: float
    anchor_max_density: float
    anchor_min_spacing: int
    anchor_mixed_axis_probability: float
    anchor_teacher_bank_mebibytes: float
    anchor_loss_weight: float
    connectivity_weight: float
    normal_transition_weight: float
    connectivity_replay_triplets_per_axis: int
    connectivity_replay_capacity_per_axis: int
    connectivity_max_triplets_per_step: int
    vf_loss_weight: float
    cfg_drop_each_probability: float
    cfg_single_drop_probability: float
    latent_channels: int
    amp_enabled: bool
    domain_dropout: float = 0.0
    anchor_pool_size: int = 4
    anchor_coarse_loss_weight: float = 0.0
    anchor_pixel_loss_weight: float = 1.0
    relation_loss_weight: float = 0.0
    relation_bank_capacity_per_axis: int = 64
    relation_profiles_per_axis: int = 4
    relation_neighbors: int = 8
    relation_quantile_low: float = 0.10
    relation_quantile_high: float = 0.90
    relation_start_step: int | None = None
    anchor_shared_axis_probability: float = 0.0
    relation_support_max_radius: int = 2
    relation_phase_weight: float = 0.75
    relation_support_weight: float = 0.25
    relation_direction_reduction: Literal["mean", "max"] = "mean"

    def __post_init__(self) -> None:
        self._validate_positive_integers()
        self._validate_non_negative_integers()
        self._validate_probabilities()
        if 3.0 * self.cfg_drop_each_probability > 1.0:
            raise ValueError(
                "the three two-condition dropout states must have total "
                "probability at most one."
            )
        if (
            not isinstance(self.anchor_teacher_bank_mebibytes, (int, float))
            or isinstance(self.anchor_teacher_bank_mebibytes, bool)
            or not math.isfinite(self.anchor_teacher_bank_mebibytes)
            or self.anchor_teacher_bank_mebibytes <= 0.0
        ):
            raise ValueError("anchor_teacher_bank_mebibytes must be positive.")
        if (
            not isinstance(self.ema_decay, (int, float))
            or isinstance(self.ema_decay, bool)
            or not math.isfinite(self.ema_decay)
            or not 0.0 <= self.ema_decay < 1.0
        ):
            raise ValueError("ema_decay must be between zero and one, excluding one.")
        self._validate_non_negative_values()
        if not isinstance(self.amp_enabled, bool):
            raise TypeError("amp_enabled must be a boolean.")
        if self.relation_start_step is not None and (
            not isinstance(self.relation_start_step, int)
            or isinstance(self.relation_start_step, bool)
            or self.relation_start_step < 0
        ):
            raise ValueError("relation_start_step must be a non-negative integer.")
        if self.relation_quantile_low >= self.relation_quantile_high:
            raise ValueError("relation quantiles must be strictly increasing.")
        if self.relation_neighbors < 2:
            raise ValueError("relation neighbors must be at least two.")
        if self.relation_neighbors > self.relation_bank_capacity_per_axis:
            raise ValueError("relation neighbors must not exceed bank capacity.")
        if (
            self.relation_loss_weight > 0.0
            and self.anchor_training_probability > 0.0
            and self.relation_start_step is not None
            and self.relation_start_step > self.anchor_start_step
        ):
            raise ValueError(
                "relation_start_step must not follow anchor_start_step when "
                "relation learning is enabled."
            )
        if self.relation_phase_weight + self.relation_support_weight <= 0.0:
            raise ValueError("at least one relation component weight must be positive.")
        if self.relation_direction_reduction not in ("mean", "max"):
            raise ValueError("relation_direction_reduction must be 'mean' or 'max'.")

    def validate_denoiser(self, denoiser: Denoiser3D) -> None:
        downsample_factor = getattr(denoiser, "downsample_factor", None)
        if (
            not isinstance(downsample_factor, int)
            or isinstance(downsample_factor, bool)
            or downsample_factor < 1
        ):
            raise ValueError("denoiser.downsample_factor must be a positive integer.")
        if self.patch_size % downsample_factor:
            raise ValueError(
                "patch_size must be divisible by the denoiser downsample factor."
            )

    def _validate_positive_integers(self) -> None:
        values = {
            "volume_batch_size": self.volume_batch_size,
            "num_phases": self.num_phases,
            "patch_size": self.patch_size,
            "slice_pairs_per_axis": self.slice_pairs_per_axis,
            "r1_interval": self.r1_interval,
            "connectivity_replay_triplets_per_axis": (
                self.connectivity_replay_triplets_per_axis
            ),
            "connectivity_replay_capacity_per_axis": (
                self.connectivity_replay_capacity_per_axis
            ),
            "connectivity_max_triplets_per_step": (
                self.connectivity_max_triplets_per_step
            ),
            "anchor_min_spacing": self.anchor_min_spacing,
            "latent_channels": self.latent_channels,
            "anchor_pool_size": self.anchor_pool_size,
            "relation_bank_capacity_per_axis": (
                self.relation_bank_capacity_per_axis
            ),
            "relation_profiles_per_axis": self.relation_profiles_per_axis,
            "relation_neighbors": self.relation_neighbors,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    def _validate_non_negative_integers(self) -> None:
        values = {
            "anchor_start_step": self.anchor_start_step,
            "anchor_ramp_steps": self.anchor_ramp_steps,
            "relation_support_max_radius": self.relation_support_max_radius,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    def _validate_probabilities(self) -> None:
        values = {
            "anchor_training_probability": self.anchor_training_probability,
            "cfg_drop_each_probability": self.cfg_drop_each_probability,
            "cfg_single_drop_probability": self.cfg_single_drop_probability,
            "anchor_multi_probability": self.anchor_multi_probability,
            "anchor_mixed_axis_probability": self.anchor_mixed_axis_probability,
            "domain_dropout": self.domain_dropout,
            "relation_quantile_low": self.relation_quantile_low,
            "relation_quantile_high": self.relation_quantile_high,
            "anchor_shared_axis_probability": (
                self.anchor_shared_axis_probability
            ),
        }
        for name, value in values.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one.")

    def _validate_non_negative_values(self) -> None:
        values = {
            "r1_gamma": self.r1_gamma,
            "critic_local_weight": self.critic_local_weight,
            "anchor_loss_weight": self.anchor_loss_weight,
            "anchor_coarse_loss_weight": self.anchor_coarse_loss_weight,
            "anchor_pixel_loss_weight": self.anchor_pixel_loss_weight,
            "relation_loss_weight": self.relation_loss_weight,
            "relation_phase_weight": self.relation_phase_weight,
            "relation_support_weight": self.relation_support_weight,
            "connectivity_weight": self.connectivity_weight,
            "normal_transition_weight": self.normal_transition_weight,
            "vf_loss_weight": self.vf_loss_weight,
        }
        for name, value in values.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")


class Trainer:
    def __init__(
        self,
        components: TrainerComponents,
        settings: TrainerSettings,
    ) -> None:
        settings.validate_denoiser(components.denoiser)
        self.denoiser = components.denoiser
        self.ema_denoiser = components.ema_denoiser
        self.critics = components.critics
        self.connectivity_critic = components.connectivity_critic
        self.streams = components.streams
        self.num_domains = len(components.streams)
        self.active_axes = tuple(int(axis) for axis in components.critics)
        self.axis_domains = {
            axis: tuple(
                domain for domain, streams in self.streams.items() if axis in streams
            )
            for axis in self.active_axes
        }
        self.diffusion = components.diffusion
        self.denoiser_optim = components.denoiser_optim
        self.critic_optims = components.critic_optims
        self.connectivity_optim = components.connectivity_optim
        self.scaler = components.scaler
        self.device = components.device
        self.volume_batch_size = settings.volume_batch_size
        self.num_phases = settings.num_phases
        self.patch_size = settings.patch_size
        self.slice_pairs_per_axis = settings.slice_pairs_per_axis
        self.ema_decay = settings.ema_decay
        self.r1_gamma = settings.r1_gamma
        self.r1_interval = settings.r1_interval
        self.critic_local_weight = settings.critic_local_weight
        self.anchor_training_probability = float(settings.anchor_training_probability)
        self.anchor_start_step = settings.anchor_start_step
        self.anchor_ramp_steps = settings.anchor_ramp_steps
        self.anchor_effective_start_step: int | None = (
            settings.anchor_start_step
            if settings.relation_loss_weight <= 0.0
            else None
        )
        self.anchor_multi_start_step = (
            settings.anchor_start_step + settings.anchor_ramp_steps
        )
        self.anchor_multi_probability = float(settings.anchor_multi_probability)
        self.anchor_shared_axis_probability = float(
            settings.anchor_shared_axis_probability
        )
        self.anchor_loss_weight = settings.anchor_loss_weight
        self.anchor_pool_size = settings.anchor_pool_size
        self.anchor_coarse_loss_weight = settings.anchor_coarse_loss_weight
        self.anchor_pixel_loss_weight = settings.anchor_pixel_loss_weight
        self.connectivity_weight = settings.connectivity_weight
        self.normal_transition_weight = settings.normal_transition_weight
        self.connect = Connectivity(
            num_phases=settings.num_phases,
            # The extra replay/teacher bank stores domain-dropped predictions.
            num_domains=self.num_domains + 1,
            patch_size=settings.patch_size,
            replay_triplets_per_axis=settings.connectivity_replay_triplets_per_axis,
            replay_capacity_per_axis=settings.connectivity_replay_capacity_per_axis,
            max_triplets_per_step=settings.connectivity_max_triplets_per_step,
            teacher_bank_bytes=round(settings.anchor_teacher_bank_mebibytes * 1024**2),
            teacher_min_entries=TEACHER_MIN_ENTRIES,
            max_density=settings.anchor_max_density,
            min_spacing=settings.anchor_min_spacing,
            mixed_axis_probability=settings.anchor_mixed_axis_probability,
        )
        self.relation_loss_weight = settings.relation_loss_weight
        self.relation_start_step = (
            self.anchor_start_step
            if settings.relation_start_step is None
            else settings.relation_start_step
        )
        self.relation_owned_axes = {
            domain: tuple(streams)
            for domain, streams in self.streams.items()
        }
        self.relation_prior_snapshot_active = False
        self.relation = RelationBank(
            num_domains=self.num_domains,
            num_phases=self.num_phases,
            axes=self.active_axes,
            capacity_per_axis=settings.relation_bank_capacity_per_axis,
            profiles_per_axis=settings.relation_profiles_per_axis,
            neighbors=settings.relation_neighbors,
            quantile_low=settings.relation_quantile_low,
            quantile_high=settings.relation_quantile_high,
            support_max_radius=settings.relation_support_max_radius,
            phase_weight=settings.relation_phase_weight,
            support_weight=settings.relation_support_weight,
            direction_reduction=settings.relation_direction_reduction,
        )
        self.vf_loss_weight = settings.vf_loss_weight
        self.cfg_drop_each_probability = float(settings.cfg_drop_each_probability)
        self.cfg_single_drop_probability = float(settings.cfg_single_drop_probability)
        self.domain_dropout = float(settings.domain_dropout)
        self.latent_channels = settings.latent_channels
        self.amp_enabled = settings.amp_enabled
        self.critic_augment = (
            CriticAugment(False)
            if components.critic_augment is None
            else components.critic_augment
        )
        self.target_count = 0
        self.target_mean = torch.zeros(settings.num_phases, dtype=torch.float64)
        self.target_m2 = torch.zeros(settings.num_phases, dtype=torch.float64)

    def step(
        self,
        step: int,
        transition: int | None = None,
    ) -> Metrics:
        prepared = self.prepare_step(step, transition)
        volume_size = self.patch_size
        transition = prepared.transition
        real = prepared.real
        critic_domains = prepared.critic_domains
        anchor = prepared.anchor
        target_vf = prepared.target_vf
        presence = prepared.presence
        ramp = prepared.anchor_ramp
        self.record_relation_reference(
            step=step,
            prepared=prepared,
        )
        (
            previous,
            current,
            logits,
            prediction,
        ) = self.generate_pair(
            transition,
            prepared.model_conditions,
            volume_size,
        )
        clean_probs = (prediction + 1.0) * 0.5
        visible_axis_masks = (
            None
            if anchor is None
            else anchor.axis_masks
            & presence.anchor.reshape(-1, 1, 1, 1, 1)
        )
        fake = {
            axis: self.critic_augment.apply_pair(
                *self.sample_pairs(
                    previous,
                    current,
                    axis,
                    axis_masks=visible_axis_masks,
                    crop_shape=tuple(real[axis].shape[-2:]),
                ),
            )
            for axis in self.active_axes
        }

        connectivity_real, connectivity_fake = self.make_connectivity_triplets(
            prediction,
            anchor,
            transition,
            self.replay_domain(prepared.model_domain),
            presence.anchor,
        )

        (
            critic_vals,
            r1,
            critic_global,
            critic_local,
        ) = self.update_critics(
            transition,
            fake,
            real,
            step,
            critic_domains,
        )
        if self.connectivity_weight > 0.0:
            critic_connectivity, connectivity_r1 = self.update_connectivity_critic(
                connectivity_real.values,
                connectivity_fake,
                step,
                prepared.model_domain,
            )
        else:
            critic_connectivity, connectivity_r1 = 0.0, 0.0
        denoiser_update = self.update_denoiser(
            DenoiserBatch(
                transition=transition,
                model_domain=prepared.model_domain,
                critic_domains=critic_domains,
                fake=fake,
                connectivity_real=connectivity_real,
                connectivity_fake=connectivity_fake,
                logits=logits,
                clean_probs=clean_probs,
                anchor=anchor,
                anchor_present=presence.anchor,
                anchor_ramp=ramp,
                target_vf=target_vf,
                vf_present=presence.vf,
            )
        )
        return self.finish_step(
            prepared=prepared,
            denoiser_update=denoiser_update,
            critic_vals=critic_vals,
            r1=r1,
            critic_global=critic_global,
            critic_local=critic_local,
            critic_connectivity=critic_connectivity,
            connectivity_r1=connectivity_r1,
            connectivity_fake=connectivity_fake,
            clean_probs=clean_probs,
            prediction=prediction,
        )

    def prepare_step(
        self,
        step: int,
        transition: int | None,
    ) -> StepPreparation:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer.")
        self.denoiser.train()
        self.critics.train()
        self.connectivity_critic.train()
        if transition is not None and (
            not isinstance(transition, int)
            or isinstance(transition, bool)
            or not 0 <= transition < self.diffusion.timesteps
        ):
            raise ValueError("transition is outside the diffusion schedule.")

        domain = self.sample_target_domain()
        model_domain = self.sample_domain_condition(domain)
        batch_domains = self.select_batch_domains(domain)
        batches = self.get_batches(domain, batch_domains)
        own_batches = {axis: batches[axis] for axis in self.streams[domain]}
        critic_domains = self.make_critic_domains(
            domain,
            model_domain,
            batch_domains,
        )
        if self.relation_loss_weight > 0.0:
            for axis, images in batches.items():
                self.relation.observe_shape(
                    axis=axis,
                    height=min(self.patch_size, images.shape[-2]),
                    width=min(self.patch_size, images.shape[-1]),
                    volume_size=self.patch_size,
                )
        vf_pool = self.get_vf_pool(own_batches)
        ramp = self.get_anchor_ramp(step)
        selection = (
            None
            if ramp == 0.0
            else self.sample_anchor(
                batches,
                self.patch_size,
                step,
                self.replay_domain(model_domain),
                owned_axes=tuple(own_batches),
            )
        )
        if (
            selection is not None
            and selection.source == "shared"
            and not self.anchor_has_compatible_vf(selection.condition, vf_pool)
        ):
            selection = self.sample_real_anchor(
                own_batches,
                self.patch_size,
                owned_axes=tuple(own_batches),
            )
        anchor = None if selection is None else selection.condition
        target_vf, resample_rate = self.resolve_target_vf(
            selection,
            vf_pool,
            anchor,
        )
        presence = self.sample_condition_presence(anchor is not None)
        model_conditions = self.make_model_conditions(
            anchor,
            target_vf,
            presence,
            self.make_domain(model_domain, self.volume_batch_size),
        )
        if transition is None:
            transition = self.sample_transition(anchor is not None)
        return StepPreparation(
            transition=transition,
            domain=domain,
            model_domain=model_domain,
            critic_domains=critic_domains,
            real=batches,
            selection=selection,
            anchor=anchor,
            target_vf=target_vf,
            presence=presence,
            model_conditions=model_conditions,
            anchor_ramp=ramp,
            vf_target_resample_rate=resample_rate,
        )

    def finish_step(
        self,
        *,
        prepared: StepPreparation,
        denoiser_update: DenoiserUpdate,
        critic_vals: list[float],
        r1: float,
        critic_global: float,
        critic_local: float,
        critic_connectivity: float,
        connectivity_r1: float,
        connectivity_fake: TripletBatch,
        clean_probs: torch.Tensor,
        prediction: torch.Tensor,
    ) -> Metrics:
        target_values, soft_values, hard_values, hard_mae = self.summarize_vfs(
            clean_probs,
            prepared.target_vf,
            prepared.presence.vf,
        )
        target_stds = self.update_target_stats(prepared.target_vf)
        self.record_connectivity_prediction(
            prediction,
            prepared.selection,
            prepared.transition,
            self.replay_domain(prepared.model_domain),
            prepared.presence.anchor,
        )
        if not self.relation_prior_snapshot_active:
            update_ema(self.ema_denoiser, self.denoiser, self.ema_decay)
        anchor = prepared.anchor
        presence = prepared.presence
        return Metrics(
            generator=denoiser_update.adversarial,
            generator_total=denoiser_update.total,
            critic=sum(critic_vals),
            r1=r1,
            transition=prepared.transition,
            volume_size=self.patch_size,
            domain=prepared.domain,
            critic_axes=tuple(critic_vals),
            anchor_planes=0 if anchor is None else anchor.planes,
            anchor_conflict_rate=0.0 if anchor is None else anchor.conflict_rate,
            anchor_loss=denoiser_update.anchor,
            anchor_accuracy=denoiser_update.anchor_accuracy,
            generator_connectivity=denoiser_update.connectivity,
            critic_connectivity=critic_connectivity,
            connectivity_r1=connectivity_r1,
            anchor_ramp=prepared.anchor_ramp,
            connectivity_triplets=len(connectivity_fake),
            connectivity_replay=self.connect.replay_size,
            anchor_teacher=(
                prepared.selection is not None
                and prepared.selection.source == "teacher"
                and bool(presence.anchor.any())
            ),
            teacher_volumes=self.connect.teacher_count,
            teacher_mebibytes=self.connect.teacher_storage_bytes / 1024**2,
            generator_global=denoiser_update.global_loss,
            generator_local=denoiser_update.local_loss,
            critic_global=critic_global,
            critic_local=critic_local,
            vf_loss=denoiser_update.vf,
            vf_active=bool(presence.vf.any()),
            target_vfs=target_values,
            target_vf_stds=target_stds,
            soft_vfs=soft_values,
            hard_vfs=hard_values,
            hard_vf_mae=hard_mae,
            vf_target_resample_rate=prepared.vf_target_resample_rate,
            anchor_input_active_fraction=float(
                presence.anchor.to(torch.float32).mean()
            ),
            vf_active_fraction=float(presence.vf.to(torch.float32).mean()),
            condition_state_fractions=presence.fractions(),
            normal_transition_loss=denoiser_update.normal_transition,
            anchor_coarse_loss=denoiser_update.anchor_coarse,
            anchor_pixel_loss=denoiser_update.anchor_pixel,
            relation_loss=denoiser_update.relation,
            relation_weighted_loss=denoiser_update.relation_weighted,
            relation_time_weight=denoiser_update.relation_time_weight,
            relation_phase_loss=denoiser_update.relation_phase,
            relation_support_loss=denoiser_update.relation_support,
            relation_minus_loss=denoiser_update.relation_minus,
            relation_plus_loss=denoiser_update.relation_plus,
            relation_queries=denoiser_update.relation_queries,
            relation_matches=denoiser_update.relation_matches,
            relation_domain_matches=denoiser_update.relation_domain_matches,
            relation_shared_matches=denoiser_update.relation_shared_matches,
            relation_ood_rejections=denoiser_update.relation_ood_rejections,
            relation_missing_references=(
                denoiser_update.relation_missing_references
            ),
            relation_distance_weights=(
                denoiser_update.relation_distance_weights
            ),
            relation_bank_entries=self.relation.entry_count,
            relation_ready_buckets=self.relation.ready_bucket_count,
            relation_prior_ready=self.relation_prior_ready(),
            anchor_shared=(
                prepared.selection is not None
                and prepared.selection.source == "shared"
                and bool(presence.anchor.any())
            ),
        )

    def resolve_target_vf(
        self,
        selection: AnchorSelection | None,
        pool: torch.Tensor,
        anchor: AnchorCondition | None,
    ) -> tuple[torch.Tensor, float]:
        if selection is not None and selection.target_vf is not None:
            return self.validate_target_vf(selection.target_vf), 0.0
        return self.sample_target_vf(pool, anchor)

    def make_connectivity_triplets(
        self,
        prediction: torch.Tensor,
        anchor: AnchorCondition | None,
        transition: int,
        domain: int,
        visible: torch.Tensor | None = None,
    ) -> tuple[TripletBatch, TripletBatch]:
        empty = TripletBatch(
            values=prediction.new_empty(
                (0, 3, self.num_phases, self.patch_size, self.patch_size)
            ),
            axes=torch.empty(0, device=self.device, dtype=torch.long),
            center_slots=torch.empty(0, device=self.device, dtype=torch.long),
        )
        enabled = self.connectivity_weight > 0.0 or self.normal_transition_weight > 0.0
        if not enabled or transition != 0 or anchor is None:
            return empty, empty
        anchor = self.visible_anchor(anchor, visible)
        if not bool(anchor.mask.any()):
            return empty, empty

        real, fake = self.connect.match_anchor(prediction, anchor, domain)
        real_values, fake_values = self.critic_augment.apply_together(
            (real.values, fake.values),
        )
        return (
            TripletBatch(
                values=real_values,
                axes=real.axes,
                center_slots=real.center_slots,
            ),
            TripletBatch(
                values=fake_values,
                axes=fake.axes,
                center_slots=fake.center_slots,
            ),
        )

    def record_connectivity_prediction(
        self,
        prediction: torch.Tensor,
        selection: AnchorSelection | None,
        transition: int,
        domain: int,
        visible: torch.Tensor | None = None,
    ) -> None:
        if transition != 0:
            return
        if selection is None:
            if self.connectivity_weight > 0.0 or self.normal_transition_weight > 0.0:
                self.connect.record_unconditional(prediction, domain)
            return
        if visible is None:
            visible = torch.ones(
                prediction.shape[0],
                device=prediction.device,
                dtype=torch.bool,
            )
        if visible.shape != (prediction.shape[0],) or visible.dtype != torch.bool:
            raise ValueError("anchor visibility must be boolean with shape [B].")
        hidden = ~visible
        if bool(hidden.any()) and (
            self.connectivity_weight > 0.0 or self.normal_transition_weight > 0.0
        ):
            self.connect.record_unconditional(prediction[hidden], domain)
        if (
            selection.source == "real"
            and self.anchor_multi_probability > 0.0
            and bool(visible.any())
        ):
            indices = visible.nonzero().flatten()
            seeds = tuple(
                selection.seeds[index]
                for index in indices.tolist()
            )
            self.connect.record_seeded(
                prediction.index_select(0, indices),
                seeds,
                domain,
            )

    @staticmethod
    def visible_anchor(
        anchor: AnchorCondition,
        visible: torch.Tensor | None,
    ) -> AnchorCondition:
        if visible is None:
            return anchor
        if visible.shape != (anchor.mask.shape[0],) or visible.dtype != torch.bool:
            raise ValueError("anchor visibility must be boolean with shape [B].")
        if visible.device != anchor.mask.device:
            raise ValueError("anchor visibility and condition must share a device.")
        mask = visible.reshape(-1, 1, 1, 1, 1)
        return AnchorCondition(
            image=anchor.image * mask,
            mask=anchor.mask & mask,
            axis_masks=anchor.axis_masks & mask,
            target=anchor.target,
            planes=anchor.planes,
            conflicts=anchor.conflicts,
            source_voxels=anchor.source_voxels,
        )

    def sample_transition(self, anchored: bool) -> int:
        if not isinstance(anchored, bool):
            raise TypeError("anchored must be a boolean.")
        if not anchored:
            return int(torch.randint(self.diffusion.timesteps, ()).item())
        if self.diffusion.timesteps == 1 or bool(torch.rand(()) < 0.25):
            return 0
        return int(torch.randint(1, self.diffusion.timesteps, ()).item())

    def select_batch_domains(self, domain: int) -> dict[int, int]:
        selected = {}
        for axis in self.active_axes:
            if axis in self.streams[domain]:
                selected[axis] = domain
                continue
            candidates = self.axis_domains[axis]
            index = int(torch.randint(len(candidates), ()).item())
            selected[axis] = candidates[index]
        return selected

    def sample_target_domain(self) -> int:
        return int(torch.randint(self.num_domains, ()).item())

    def get_batches(
        self,
        domain: int,
        batch_domains: dict[int, int] | None = None,
    ) -> dict[int, torch.Tensor]:
        if batch_domains is None:
            batch_domains = self.select_batch_domains(domain)
        return {
            axis: self.streams[batch_domains[axis]][axis]
            .next()
            .to(
                self.device,
                non_blocking=True,
            )
            for axis in self.active_axes
        }

    def sample_domain_condition(self, domain: int) -> int:
        if self.domain_dropout > 0.0 and bool(torch.rand(()) < self.domain_dropout):
            return NULL_DOMAIN
        return domain

    @staticmethod
    def make_critic_domains(
        domain: int,
        model_domain: int,
        batch_domains: dict[int, int],
    ) -> dict[int, int]:
        if model_domain == NULL_DOMAIN:
            return {axis: NULL_DOMAIN for axis in batch_domains}
        return {
            axis: domain if source_domain == domain else NULL_DOMAIN
            for axis, source_domain in batch_domains.items()
        }

    def replay_domain(self, model_domain: int) -> int:
        return self.num_domains if model_domain == NULL_DOMAIN else model_domain

    def make_domain(self, domain: int, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            domain,
            device=self.device,
            dtype=torch.long,
        )

    @staticmethod
    def crop_images(
        images: torch.Tensor,
        size: int | tuple[int, int],
        centers: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        if images.ndim not in (3, 4):
            raise ValueError("images must have shape [B, H, W] or [B, C, H, W].")
        if isinstance(size, int) and not isinstance(size, bool):
            crop_h = crop_w = size
        elif (
            isinstance(size, tuple)
            and len(size) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in size
            )
        ):
            crop_h, crop_w = size
        else:
            raise ValueError("crop size must be a positive integer.")
        if crop_h < 1 or crop_w < 1:
            raise ValueError("crop size must be a positive integer.")
        height, width = images.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError("crop size must fit inside the images.")
        if (height, width) == (crop_h, crop_w):
            return images

        top = torch.randint(height - crop_h + 1, (images.shape[0],)).tolist()
        left = torch.randint(width - crop_w + 1, (images.shape[0],)).tolist()
        if centers is not None:
            if len(centers) > images.shape[0]:
                raise ValueError("centers must not outnumber images.")
            for index, (row, col) in enumerate(centers):
                top[index] = min(max(row - crop_h // 2, 0), height - crop_h)
                left[index] = min(max(col - crop_w // 2, 0), width - crop_w)
        return torch.stack(
            [
                image[..., row : row + crop_h, col : col + crop_w]
                for image, row, col in zip(images, top, left, strict=True)
            ]
        )

    def get_vf_pool(self, batches: dict[int, torch.Tensor]) -> torch.Tensor:
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
            if int(lower) < 0 or int(upper) >= self.num_phases:
                raise ValueError("training images contain a phase outside num_phases.")
            batch = labels.shape[0]
            offsets = torch.arange(
                batch,
                device=labels.device,
                dtype=labels.dtype,
            ).mul_(self.num_phases)
            encoded = labels.flatten(1).add(offsets[:, None])
            counts = torch.bincount(
                encoded.flatten(),
                minlength=batch * self.num_phases,
            ).reshape(batch, self.num_phases)
            fractions = counts.to(torch.float32).div_(
                labels.shape[-2] * labels.shape[-1]
            )
            values.append(fractions)
        return torch.cat(values, dim=0)

    def validate_target_vf(self, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(target, torch.Tensor) or not target.is_floating_point():
            raise TypeError("target VF must be a floating-point tensor.")
        expected = (self.volume_batch_size, self.num_phases)
        if target.shape != expected:
            raise ValueError(f"target VF must have shape {expected}.")
        target = target.to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(target).all()) or bool((target < 0.0).any()):
            raise ValueError("target VF values must be finite and non-negative.")
        sums = target.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=0.0):
            raise ValueError("target VF rows must sum to one.")
        return target

    def sample_target_vf(
        self,
        pool: torch.Tensor,
        anchor: AnchorCondition | None,
    ) -> tuple[torch.Tensor, float]:
        if (
            not isinstance(pool, torch.Tensor)
            or not pool.is_floating_point()
            or pool.ndim != 2
            or pool.shape[0] < 1
            or pool.shape[1] != self.num_phases
        ):
            raise ValueError("VF pool must have shape [N, num_phases].")
        pool = pool.to(device=self.device, dtype=torch.float32)
        indices = torch.randint(
            pool.shape[0],
            (self.volume_batch_size,),
            device=self.device,
        )
        resampled = 0
        if anchor is not None:
            if anchor.target.shape[0] != self.volume_batch_size:
                raise ValueError("anchor and generated volume batches must match.")
            required = self.anchor_minimum_vfs(anchor)
            for batch, minimum in enumerate(required):
                valid = (pool + 1e-7 >= minimum).all(dim=1).nonzero().flatten()
                if not len(valid):
                    raise ValueError(
                        "anchor phase minima are incompatible with every empirical "
                        "VF target."
                    )
                if not bool((pool[indices[batch]] + 1e-7 >= minimum).all()):
                    choice = int(torch.randint(len(valid), (), device=self.device))
                    indices[batch] = valid[choice]
                    resampled += 1
        target = self.validate_target_vf(pool.index_select(0, indices))
        return target, resampled / self.volume_batch_size

    def anchor_has_compatible_vf(
        self,
        anchor: AnchorCondition,
        pool: torch.Tensor,
    ) -> bool:
        """Whether every anchor sample fits at least one target-domain VF."""
        required = self.anchor_minimum_vfs(anchor)
        pool = pool.to(device=required.device, dtype=torch.float32)
        compatible = (pool.unsqueeze(0) + 1e-7 >= required.unsqueeze(1)).all(dim=2)
        return bool(compatible.any(dim=1).all())

    def anchor_minimum_vfs(self, anchor: AnchorCondition) -> torch.Tensor:
        if anchor.target.shape[0] != self.volume_batch_size:
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
                minlength=self.num_phases,
            ).to(torch.float32)
            required.append(counts / voxel_count)
        return torch.stack(required)

    def sample_condition_presence(self, has_anchor: bool) -> ConditionPresence:
        if not isinstance(has_anchor, bool):
            raise TypeError("has_anchor must be a boolean.")
        batch = self.volume_batch_size
        anchor = torch.full(
            (batch,),
            has_anchor,
            device=self.device,
            dtype=torch.bool,
        )
        vf = torch.ones(batch, device=self.device, dtype=torch.bool)
        random = torch.rand(batch, device=self.device)
        if has_anchor:
            probability = self.cfg_drop_each_probability
            joint_null = random < probability
            anchor_null = (random >= probability) & (random < 2.0 * probability)
            vf_null = (random >= 2.0 * probability) & (random < 3.0 * probability)
            anchor = ~(joint_null | anchor_null)
            vf = ~(joint_null | vf_null)
        else:
            vf = random >= self.cfg_single_drop_probability
        return ConditionPresence(anchor=anchor, vf=vf)

    @staticmethod
    def make_model_conditions(
        anchor: AnchorCondition | None,
        target_vf: torch.Tensor,
        presence: ConditionPresence,
        domain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditions = {
            "domain": domain,
            "vf": target_vf,
            "vf_present": presence.vf,
        }
        if anchor is not None:
            visible = presence.anchor.reshape(-1, 1, 1, 1, 1)
            conditions["anchor_image"] = anchor.image
            conditions["anchor_mask"] = anchor.mask * visible.to(anchor.mask.dtype)
        return conditions

    def get_anchor_ramp(self, step: int) -> float:
        if step < self.anchor_start_step:
            return 0.0
        if not self.relation_prior_ready():
            return 0.0
        if self.anchor_effective_start_step is None:
            self.anchor_effective_start_step = step
            self.anchor_multi_start_step = step + self.anchor_ramp_steps
        start_step = self.anchor_effective_start_step
        if self.anchor_ramp_steps == 0:
            return 1.0
        return min(
            (step - start_step + 1) / self.anchor_ramp_steps,
            1.0,
        )

    def relation_prior_ready(self) -> bool:
        return self.relation_loss_weight <= 0.0 or self.relation.prior_ready(
            self.relation_owned_axes
        )

    def sample_anchor(
        self,
        batches: dict[int, torch.Tensor],
        volume_size: int,
        step: int,
        domain: int,
        *,
        owned_axes: tuple[int, ...] | None = None,
    ) -> AnchorSelection | None:
        probability = self.anchor_training_probability
        if probability <= 0.0:
            return None
        if probability < 1.0 and not bool(torch.rand(()) < probability):
            return None
        if (
            step >= self.anchor_multi_start_step
            and self.anchor_multi_probability > 0.0
            and bool(torch.rand(()) < self.anchor_multi_probability)
        ):
            teacher = self.connect.sample_teacher(
                domain=domain,
                volume_size=volume_size,
                batch_size=self.volume_batch_size,
                device=self.device,
                dtype=torch.float32,
            )
            if teacher is not None:
                return AnchorSelection(
                    condition=teacher.condition,
                    source="teacher",
                    target_vf=teacher.target_vf,
                )
        return self.sample_real_anchor(
            batches,
            volume_size,
            owned_axes=owned_axes,
        )

    def sample_real_anchor(
        self,
        batches: dict[int, torch.Tensor],
        volume_size: int,
        *,
        owned_axes: tuple[int, ...] | None = None,
    ) -> AnchorSelection:
        if not batches:
            raise ValueError("anchor batches must not be empty.")
        if owned_axes is None:
            owned_axes = tuple(batches)
        if not owned_axes or any(axis not in batches for axis in owned_axes):
            raise ValueError("owned anchor axes must be present in the batches.")
        shared_axes = tuple(axis for axis in batches if axis not in owned_axes)
        use_shared = (
            bool(shared_axes)
            and getattr(self, "anchor_shared_axis_probability", 0.0) > 0.0
            and bool(
                torch.rand(())
                < getattr(self, "anchor_shared_axis_probability", 0.0)
            )
        )
        axes = shared_axes if use_shared else owned_axes
        axis = axes[int(torch.randint(len(axes), ()).item())]
        images = batches[axis]
        if images.shape[0] < self.volume_batch_size:
            raise ValueError(
                "the real image batch must cover the generated volume batch."
            )
        batch_indices = torch.randperm(
            images.shape[0],
            device=images.device,
        )[: self.volume_batch_size]
        selected = images.index_select(0, batch_indices)
        shape = tuple(min(volume_size, size) for size in selected.shape[-2:])
        selected = self.crop_images(selected, shape)
        position = tuple(
            int(torch.randint(volume_size - size + 1, ()).item()) for size in shape
        )
        plane_index = int(torch.randint(volume_size, ()).item())
        plane = PlaneAnchor(
            image=selected,
            axis=axis,
            index=plane_index,
            position=position,
        )
        condition = build_anchors(
            (plane,),
            batch_size=self.volume_batch_size,
            num_phases=self.num_phases,
            volume_size=volume_size,
            device=self.device,
            dtype=torch.float32,
            reconcile=False,
        )
        if condition is None:
            raise RuntimeError("real anchor construction returned no condition.")
        seeds = tuple(
            PlaneAnchor(
                image=selected[batch],
                axis=axis,
                index=plane_index,
                position=position,
            )
            for batch in range(self.volume_batch_size)
        )
        return AnchorSelection(
            condition=condition,
            source="shared" if use_shared else "real",
            seeds=seeds,
        )

    @torch.no_grad()
    def record_relation_reference(
        self,
        *,
        step: int,
        prepared: StepPreparation,
    ) -> None:
        owned_axes = self.relation_owned_axes[prepared.domain]
        if self.relation_loss_weight <= 0.0:
            self.relation_prior_snapshot_active = False
            return
        if step < self.relation_start_step or self.relation_prior_ready():
            return
        # Keep one EMA snapshot fixed until every owned domain/axis bucket is
        # complete. This avoids a moving relation target without a second model.
        self.relation_prior_snapshot_active = True
        if not self.relation.needs_data(
            condition_domain=prepared.domain,
            source_domain=prepared.domain,
            owned_axes=owned_axes,
        ):
            return
        conditions = {
            name: values
            for name, values in prepared.model_conditions.items()
            if name not in ("anchor_image", "anchor_mask")
        }
        conditions["domain"] = self.make_domain(
            prepared.domain,
            self.volume_batch_size,
        )
        conditions["vf"] = torch.zeros_like(prepared.target_vf)
        conditions["vf_present"] = torch.zeros_like(prepared.presence.vf)
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            shape = (
                self.volume_batch_size,
                self.num_phases,
                self.patch_size,
                self.patch_size,
                self.patch_size,
            )
            initial = torch.randn(shape, device=self.device, dtype=torch.float32)
            with self.autocast():
                prediction = self.diffusion.sample(
                    self.ema_denoiser,
                    initial,
                    self.latent_channels,
                    conditions,
                )
            probs = (prediction.float() + 1.0) * 0.5
            self.relation.add(
                probs,
                condition_domain=prepared.domain,
                source_domain=prepared.domain,
                owned_axes=owned_axes,
            )
        if self.relation_prior_ready():
            self.relation_prior_snapshot_active = False

    def generate_pair(
        self,
        transition: int,
        model_conditions: dict[str, torch.Tensor],
        volume_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (
            self.volume_batch_size,
            self.num_phases,
            volume_size,
            volume_size,
            volume_size,
        )
        current = torch.randn(shape, device=self.device, dtype=torch.float32)

        with torch.no_grad(), self.autocast():
            for index in reversed(range(transition + 1, self.diffusion.timesteps)):
                time = self.make_time(index, current.shape[0])
                latent = self.sample_latent(current.shape[0], current.dtype)
                prediction = self.denoiser(
                    current,
                    time,
                    latent,
                    **model_conditions,
                )
                current = self.diffusion.sample_posterior(
                    current,
                    prediction,
                    index,
                )

        current = current.detach()
        time = self.make_time(transition, current.shape[0])
        latent = self.sample_latent(current.shape[0], current.dtype)
        with self.autocast():
            logits = self.denoiser.predict_logits(
                current,
                time,
                latent,
                **model_conditions,
            )
            prediction = self.denoiser.decode(logits)
            previous = self.diffusion.sample_posterior(
                current,
                prediction,
                transition,
            )
        return previous, current, logits, prediction

    def sample_pairs(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        axis: int,
        axis_masks: torch.Tensor | None = None,
        crop_shape: int | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous.shape != current.shape:
            raise ValueError("previous and current volumes must have the same shape.")
        if previous.ndim != 5:
            raise ValueError("volumes must have shape [B, C, D, H, W].")
        if axis not in AXES:
            raise ValueError("axis must be 0, 1, or 2.")
        if axis_masks is not None and (
            axis_masks.shape != (previous.shape[0], 3, *previous.shape[2:])
            or axis_masks.device != previous.device
        ):
            raise ValueError(
                "axis_masks must match the volume batch and spatial shape."
            )

        count = self.slice_pairs_per_axis
        batch_indices = torch.randint(
            previous.shape[0],
            (count,),
            device=previous.device,
        )
        plane_indices = torch.randint(
            previous.shape[axis + 2],
            (count,),
            device=previous.device,
        )
        focused = 0
        centers: list[tuple[int, int]] = []
        if axis_masks is not None:
            normals = tuple(normal for normal in AXES if normal != axis)
            focus = axis_masks[:, normals].any(dim=1, keepdim=True)
            focus = focus.movedim(axis + 2, 2)[:, 0]
            points = focus.nonzero()
            if points.numel():
                focused = min(count, max(1, count // 2))
                selected = points.index_select(
                    0,
                    torch.randint(points.shape[0], (focused,), device=points.device),
                )
                batch_indices[:focused] = selected[:, 0]
                plane_indices[:focused] = selected[:, 1]
                centers = [
                    (int(row), int(col)) for row, col in selected[:, 2:].tolist()
                ]
        previous = previous.movedim(axis + 2, 2)
        current = current.movedim(axis + 2, 2)
        previous = previous[batch_indices, :, plane_indices]
        current = current[batch_indices, :, plane_indices]
        channels = previous.shape[1]
        crop_shape = self.patch_size if crop_shape is None else crop_shape
        pairs = self.crop_images(
            torch.cat((previous, current), dim=1),
            crop_shape,
            centers,
        )
        return pairs[:, :channels], pairs[:, channels:]

    def update_critics(
        self,
        transition: int,
        fake: dict[int, tuple[torch.Tensor, torch.Tensor]],
        batches: dict[int, torch.Tensor],
        step: int,
        domains: dict[int, int],
    ) -> tuple[list[float], float, float, float]:
        apply_r1 = self.r1_gamma > 0.0 and (step + 1) % self.r1_interval == 0
        critic_losses = [0.0] * len(AXES)
        r1_sum = 0.0
        global_sum = 0.0
        local_sum = 0.0
        local_weight = self.critic_local_weight
        for axis in self.active_axes:
            critic = self.critics[str(axis)]
            optimizer = self.critic_optims[str(axis)]
            optimizer.zero_grad(set_to_none=True)

            images = batches[axis]
            real = (
                F.one_hot(images, num_classes=self.num_phases)
                .movedim(-1, 1)
                .to(torch.float32)
                .mul_(2.0)
                .sub_(1.0)
            )
            real_time = self.make_time(transition, real.shape[0])
            real_prev, real_curr = self.diffusion.sample_pair(
                real,
                transition,
            )
            real_prev, real_curr = self.critic_augment.apply_pair(
                real_prev,
                real_curr,
            )
            real_prev.requires_grad_(apply_r1)
            fake_prev, fake_curr = fake[axis]
            fake_prev = fake_prev.detach().float()
            fake_curr = fake_curr.detach().float()
            fake_time = self.make_time(transition, fake_prev.shape[0])
            real_domain = self.make_domain(domains[axis], real_prev.shape[0])
            fake_domain = self.make_domain(domains[axis], fake_prev.shape[0])

            autocast = self.autocast(self.amp_enabled and not apply_r1)
            with autocast:
                real_score = critic(real_prev, real_curr, real_time, real_domain)
                fake_score = critic(fake_prev, fake_curr, fake_time, fake_domain)
                losses = get_critic_loss(real_score, fake_score)
                loss = losses.combine(local_weight)
            global_sum += float(losses.global_loss.detach())
            local_sum += float(losses.local_loss.detach())
            if apply_r1:
                r1 = get_critic_r1(
                    real_score,
                    (real_prev,),
                )
                penalty = r1.combine(local_weight)
                r1_sum += float(penalty.detach())
                loss = loss + 0.5 * self.r1_gamma * self.r1_interval * penalty
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            critic_losses[axis] = float(loss.detach())
        self.scaler.update()
        return critic_losses, r1_sum, global_sum, local_sum

    def update_connectivity_critic(
        self,
        real: torch.Tensor,
        fake: TripletBatch,
        step: int,
        domain: int,
    ) -> tuple[float, float]:
        if not len(fake):
            return 0.0, 0.0
        if real.shape != fake.values.shape:
            raise ValueError("real and fake connectivity triplets must match.")

        apply_r1 = self.r1_gamma > 0.0 and (step + 1) % self.r1_interval == 0
        self.connectivity_optim.zero_grad(set_to_none=True)
        real = real.detach().float().requires_grad_(apply_r1)
        fake_values = fake.values.detach().float()
        domains = self.make_domain(domain, len(fake))
        autocast = self.autocast(self.amp_enabled and not apply_r1)
        with autocast:
            real_score = self.connectivity_critic(real, fake.axes, domains)
            fake_score = self.connectivity_critic(fake_values, fake.axes, domains)
            losses = get_critic_loss(real_score, fake_score)
            loss = losses.combine(self.critic_local_weight)
        adversarial = float(loss.detach())
        r1_value = 0.0
        if apply_r1:
            r1 = get_critic_r1(real_score, (real,))
            penalty = r1.combine(self.critic_local_weight)
            r1_value = float(penalty.detach())
            loss = loss + 0.5 * self.r1_gamma * self.r1_interval * penalty
        self.scaler.scale(loss).backward()
        self.scaler.step(self.connectivity_optim)
        self.scaler.update()
        return adversarial, r1_value

    def update_denoiser(
        self,
        batch: DenoiserBatch,
    ) -> DenoiserUpdate:
        self.denoiser_optim.zero_grad(set_to_none=True)
        for critic in self.critics.values():
            critic.requires_grad_(False)
        self.connectivity_critic.requires_grad_(False)
        try:
            heads = []
            local_weight = self.critic_local_weight
            with self.autocast():
                for axis in self.active_axes:
                    fake_prev, fake_curr = batch.fake[axis]
                    time = self.make_time(batch.transition, fake_prev.shape[0])
                    domains = self.make_domain(
                        batch.critic_domains[axis],
                        fake_prev.shape[0],
                    )
                    scores = self.critics[str(axis)](
                        fake_prev,
                        fake_curr,
                        time,
                        domains,
                    )
                    heads.append(get_generator_loss(scores))
                global_loss = torch.stack([loss.global_loss for loss in heads]).sum()
                local_loss = torch.stack([loss.local_loss for loss in heads]).sum()
                adversarial_loss = global_loss + local_weight * local_loss
                connectivity_loss = adversarial_loss.new_zeros(())
                if len(batch.connectivity_fake) and self.connectivity_weight > 0.0:
                    connectivity_scores = self.connectivity_critic(
                        batch.connectivity_fake.values,
                        batch.connectivity_fake.axes,
                        self.make_domain(
                            batch.model_domain,
                            len(batch.connectivity_fake),
                        ),
                    )
                    connectivity_head = get_generator_loss(connectivity_scores)
                    connectivity_loss = connectivity_head.combine(local_weight)
                normal_loss = adversarial_loss.new_zeros(())
                if len(batch.connectivity_fake) and self.normal_transition_weight > 0.0:
                    normal_loss = normal_transition_loss(
                        batch.connectivity_real,
                        batch.connectivity_fake,
                    )
                anchor_loss = adversarial_loss.new_zeros(())
                anchor_coarse = adversarial_loss.new_zeros(())
                anchor_pixel = adversarial_loss.new_zeros(())
                anchor_accuracy = adversarial_loss.new_zeros(())
                if batch.anchor is not None:
                    anchor_result = soft_anchor_loss(
                        batch.logits,
                        batch.anchor,
                        batch.anchor_present,
                        pool_size=self.anchor_pool_size,
                        coarse_weight=self.anchor_coarse_loss_weight,
                        pixel_weight=self.anchor_pixel_loss_weight,
                    )
                    anchor_loss = anchor_result.total
                    anchor_coarse = anchor_result.coarse
                    anchor_pixel = anchor_result.pixel
                    anchor_accuracy = anchor_result.accuracy
                relation_loss = adversarial_loss.new_zeros(())
                relation_queries = 0
                relation_matches = 0
                relation_domain_matches = 0
                relation_shared_matches = 0
                relation_ood_rejections = 0
                relation_missing_references = 0
                relation_phase = adversarial_loss.new_zeros(())
                relation_support = adversarial_loss.new_zeros(())
                relation_minus = adversarial_loss.new_zeros(())
                relation_plus = adversarial_loss.new_zeros(())
                relation_distance_weights = adversarial_loss.new_zeros(
                    self.patch_size - 1
                )
                if batch.anchor is not None and self.relation_loss_weight > 0.0:
                    relation_result = self.relation.loss(
                        batch.clean_probs,
                        batch.anchor,
                        batch.anchor_present,
                        domain=batch.model_domain,
                    )
                    relation_loss = relation_result.loss
                    relation_queries = relation_result.queries
                    relation_matches = relation_result.matches
                    relation_domain_matches = relation_result.domain_matches
                    relation_shared_matches = relation_result.shared_matches
                    relation_ood_rejections = relation_result.ood_rejections
                    relation_missing_references = relation_result.missing_references
                    relation_phase = relation_result.phase
                    relation_support = relation_result.support
                    relation_minus = relation_result.minus
                    relation_plus = relation_result.plus
                    relation_distance_weights = relation_result.distance_weights
                vf_loss = adversarial_loss.new_zeros(())
                if bool(batch.vf_present.any()):
                    pred_vf = batch.clean_probs.mean(dim=(2, 3, 4))
                    per_sample = 0.5 * (pred_vf - batch.target_vf).abs().sum(dim=1)
                    vf_loss = per_sample[batch.vf_present].mean()
                relation_time_weight = self.diffusion.alpha_bars[
                    batch.transition + 1
                ].to(device=adversarial_loss.device, dtype=adversarial_loss.dtype)
                relation_weighted = (
                    batch.anchor_ramp
                    * self.relation_loss_weight
                    * relation_time_weight
                    * relation_loss
                )
                total = (
                    adversarial_loss
                    + batch.anchor_ramp
                    * (
                        self.connectivity_weight * connectivity_loss
                        + self.normal_transition_weight * normal_loss
                        + self.anchor_loss_weight * anchor_loss
                    )
                    + relation_weighted
                    + self.vf_loss_weight * vf_loss
                )
            self.scaler.scale(total).backward()
            self.scaler.step(self.denoiser_optim)
            self.scaler.update()
        finally:
            for critic in self.critics.values():
                critic.requires_grad_(True)
            self.connectivity_critic.requires_grad_(True)
        return DenoiserUpdate(
            adversarial=float(adversarial_loss.detach()),
            total=float(total.detach()),
            global_loss=float(global_loss.detach()),
            local_loss=float(local_loss.detach()),
            connectivity=float(connectivity_loss.detach()),
            normal_transition=float(normal_loss.detach()),
            anchor=float(anchor_loss.detach()),
            anchor_coarse=float(anchor_coarse.detach()),
            anchor_pixel=float(anchor_pixel.detach()),
            anchor_accuracy=float(anchor_accuracy.detach()),
            relation=float(relation_loss.detach()),
            relation_weighted=float(relation_weighted.detach()),
            relation_time_weight=float(relation_time_weight.detach()),
            relation_phase=float(relation_phase.detach()),
            relation_support=float(relation_support.detach()),
            relation_minus=float(relation_minus.detach()),
            relation_plus=float(relation_plus.detach()),
            relation_queries=relation_queries,
            relation_matches=relation_matches,
            relation_domain_matches=relation_domain_matches,
            relation_shared_matches=relation_shared_matches,
            relation_ood_rejections=relation_ood_rejections,
            relation_missing_references=relation_missing_references,
            relation_distance_weights=tuple(
                float(value)
                for value in relation_distance_weights.detach().to("cpu")
            ),
            vf=float(vf_loss.detach()),
        )

    @staticmethod
    def summarize_vfs(
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

    def update_target_stats(
        self,
        target: torch.Tensor,
    ) -> tuple[float, ...]:
        values = target.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != self.num_phases:
            raise ValueError("target VF statistics require shape [B, num_phases].")
        for value in values:
            self.target_count += 1
            delta = value - self.target_mean
            self.target_mean.add_(delta / self.target_count)
            self.target_m2.add_(delta * (value - self.target_mean))
        if self.target_count < 2:
            return tuple(0.0 for _ in range(self.num_phases))
        std = (self.target_m2 / (self.target_count - 1)).sqrt()
        return tuple(float(value) for value in std)

    def make_time(self, transition: int, batch: int) -> torch.Tensor:
        return torch.full(
            (batch,),
            transition,
            device=self.device,
            dtype=torch.long,
        )

    def sample_latent(self, batch: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            batch,
            self.latent_channels,
            device=self.device,
            dtype=dtype,
        )

    def autocast(
        self,
        enabled: bool | None = None,
    ) -> AbstractContextManager:
        if enabled is None:
            enabled = self.amp_enabled
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=enabled,
        )
