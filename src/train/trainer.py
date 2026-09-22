import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from src.anchor import AnchorCondition, PlaneAnchor, encode_anchors
from src.data.augment import CriticAugment, crop_images
from src.data.loader import BatchStream
from src.data.slice import AnchorTripletSampler, TripletBatch, sample_pairs
from src.evaluate.anchor import anchor_boundary_metrics
from src.evaluate.label import compute_vf
from src.evaluate.profile import phase_profile
from src.evaluate.structure import structure_metrics
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.model.layers import NULL_DOMAIN
from src.plane import AXES, PLANE_AXES
from src.prepare.height import height_field
from src.prepare.profile import image_profile, profile_field, rebin_profile
from src.prepare.resize import resize_phases
from src.train.anchor_bank import AnchorBank
from src.train.batch import (
    AnchorSelection,
    ConditionPresence,
    DenoiserBatch,
    DenoiserUpdate,
    RealBatch,
    SampledPairs,
    StepPreparation,
)
from src.train.coarse import corrupt_coarse
from src.train.ema import update_ema
from src.train.loss.anchor import SoftAnchorLoss
from src.train.loss.connectivity import compute_transition_loss
from src.train.loss.gan import (
    HeadLoss,
    get_critic_loss,
    get_critic_r1,
    get_generator_loss,
)
from src.train.loss.spatial_profile import compute_profile_loss
from src.train.loss.sr import consistency_loss
from src.train.loss.volume_fraction import compute_vf_loss
from src.train.metrics import Metrics, materialize_metrics
from src.train.step import (
    input_gradient_norms,
    step_optimizer,
    validate_loss,
)


@dataclass(frozen=True)
class TrainerComponents:
    denoiser: Denoiser3D
    ema_denoiser: Denoiser3D
    critics: nn.ModuleDict
    connectivity_critic: nn.Module | None
    streams: dict[int, dict[int, BatchStream]]
    diffusion: Diffusion
    denoiser_optim: torch.optim.Optimizer
    critic_optims: dict[str, torch.optim.Optimizer]
    connectivity_optim: torch.optim.Optimizer | None
    scaler: torch.amp.GradScaler
    device: torch.device
    critic_augment: CriticAugment | None = None
    coarse_bank: dict | None = None
    bank_origins: dict | None = None
    bank_extents: dict | None = None
    critic_groups_by_domain: dict | None = None
    data_fingerprint: dict[str, str] = field(default_factory=dict)


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
    connectivity_weight: float
    normal_transition_weight: float
    vf_loss_weight: float
    cfg_drop_each_probability: float
    latent_channels: int
    amp_enabled: bool
    domain_dropout: float
    anchor_pixel_loss_weight: float
    anchor_shared_axis_probability: float
    connectivity_max_gap: int = 1
    r2_gamma: float = 0.0
    connectivity_start_step: int = 0
    connectivity_ramp_steps: int = 20000
    connectivity_windows_per_plane: int = 4
    anchor_bank_capacity: int = 4
    anchor_plane_spacing: int = 16
    structure_every_steps: int = 100
    consistency_weight: float = 0.0
    consistency_tolerance: float = 0.0
    coarse_corruption_probability: float = 0.0
    coarse_corruption_strength: float = 0.0
    cfg: dict = field(default_factory=dict)
    height_data: dict | None = None
    profile_settings: dict = field(
        default_factory=lambda: {"enabled": False, "num_bins": 16}
    )
    profile_weight: float = 0.0
    profile_gradient_weight: float = 0.0


class Trainer:
    def __init__(
        self,
        components: TrainerComponents,
        settings: TrainerSettings,
    ) -> None:
        self.denoiser = components.denoiser
        self.ema_denoiser = components.ema_denoiser
        self.critics = components.critics
        self.connectivity_critic = components.connectivity_critic
        self.streams = components.streams
        self.bank = components.coarse_bank
        self.bank_origins = components.bank_origins
        self.bank_extents = components.bank_extents
        self.data_fingerprint = components.data_fingerprint
        self.cfg = settings.cfg
        self.critic_groups_by_domain = components.critic_groups_by_domain
        self.critic_groups = (
            {
                group: tuple(PLANE_AXES[plane] for plane in group.split("_"))
                for group in self.critics
            }
            if self.critic_groups_by_domain is None
            else self.critic_groups_by_domain[0]
        )
        self.axis_critics = {
            axis: group for group, axes in self.critic_groups.items() for axis in axes
        }
        self.active_axes = tuple(sorted(self.axis_critics))
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
        self.anchor_shared_axis_probability = float(
            settings.anchor_shared_axis_probability
        )
        pool_exponent = math.ceil(math.log2(self.denoiser.downsample_factor) / 2.0)
        self.anchor_loss = SoftAnchorLoss(
            pool_size=2**pool_exponent,
            pixel_weight=settings.anchor_pixel_loss_weight,
        )
        self.connectivity_weight = settings.connectivity_weight
        self.normal_transition_weight = settings.normal_transition_weight
        self.connectivity_start_step = settings.connectivity_start_step
        self.connectivity_ramp_steps = settings.connectivity_ramp_steps
        self.anchor_triplets = AnchorTripletSampler(
            max_gap=settings.connectivity_max_gap,
            windows_per_plane=settings.connectivity_windows_per_plane,
        )
        self.anchor_bank = (
            None
            if self.bank is not None
            else AnchorBank(
                settings.anchor_bank_capacity, settings.anchor_plane_spacing
            )
        )
        self.structure_every_steps = settings.structure_every_steps
        self.use_multi_anchor_next = False
        self.vf_loss_weight = settings.vf_loss_weight
        self.consistency_weight = settings.consistency_weight
        self.consistency_tolerance = settings.consistency_tolerance
        self.coarse_corruption_probability = settings.coarse_corruption_probability
        self.coarse_corruption_strength = settings.coarse_corruption_strength
        self.cfg_drop_each_probability = float(settings.cfg_drop_each_probability)
        self.domain_dropout = float(settings.domain_dropout)
        self.latent_channels = settings.latent_channels
        self.amp_enabled = settings.amp_enabled
        if (
            type(self.r1_interval) is not int
            or self.r1_interval < 1
            or not math.isfinite(self.r1_gamma)
            or self.r1_gamma < 0
            or not math.isfinite(settings.r2_gamma)
            or settings.r2_gamma < 0
        ):
            raise ValueError(
                "regularization interval must be positive and R2 weight finite/non-negative."
            )
        self.r2_gamma = settings.r2_gamma
        self.updates = {
            name: 0 for name in (*self.critics, "connectivity", "generator")
        }
        self.completed_steps = 0
        self.path_maps = []
        self.height_data = settings.height_data
        self.profile_settings = settings.profile_settings
        self.profile_weight = settings.profile_weight
        self.profile_gradient_weight = settings.profile_gradient_weight
        self.diagnostics = {}
        self.generator_updated = True
        self.critic_augment = (
            CriticAugment()
            if components.critic_augment is None
            else components.critic_augment
        )

    def step(
        self,
        step: int,
        transition: int | None = None,
    ) -> Metrics:
        self.diagnostics = {}
        prepared = self.prepare_step(step, transition)
        volume_size = self.patch_size
        transition = prepared.transition
        real = prepared.real
        critic_domains = prepared.critic_domains
        presence = prepared.presence
        selection = prepared.selection
        self.diagnostics["sampling/reference_passes"] = 0
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
        anchor = None if selection is None else selection.condition
        clean_probs = (prediction + 1.0) * 0.5
        measured = None if selection is None else selection.measured
        profile = (
            prepared.model_conditions.get("profile")
            if self.profile_settings.get("critic_enabled", False)
            else None
        )
        profile_present = presence.vf.to(self.device)
        fake = self._sample_fake_pairs(previous, current, prepared)
        connectivity_real, connectivity_fake = self.make_connectivity_triplets(
            prediction,
            None if selection is None else selection.reference,
            anchor,
            transition,
            presence.anchor & presence.vf
            if self.profile_settings["enabled"]
            else presence.anchor,
            None if selection is None else selection.source,
            height=prepared.model_conditions.get("height"),
            profile=profile,
            profile_present=profile_present,
        )
        if measured is not None and transition == 0:
            self.diagnostics.update(
                anchor_boundary_metrics(
                    prediction, self.visible_anchor(measured, presence.anchor)
                )
            )
        connectivity_domains = self.get_connectivity_domains(
            prepared.critic_domains,
            connectivity_fake,
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
            profile_present=profile_present,
        )
        if self.connectivity_weight > 0.0:
            critic_connectivity, connectivity_r1 = self.update_connectivity_critic(
                connectivity_real.values,
                connectivity_fake,
                step,
                connectivity_domains,
            )
        else:
            critic_connectivity, connectivity_r1 = 0.0, 0.0
        denoiser_update = self.update_denoiser(
            self._make_denoiser_batch(
                prepared=prepared,
                connectivity_domains=connectivity_domains,
                fake=fake,
                connectivity_real=connectivity_real,
                connectivity_fake=connectivity_fake,
                logits=logits,
                clean_probs=clean_probs,
            )
        )
        if (
            self.generator_updated
            and transition == 0
            and selection is not None
            and selection.source in ("real", "shared")
            and measured is not None
        ):
            self.anchor_bank.add(
                prepared.domain,
                prediction,
                measured,
                presence.anchor & presence.vf
                if self.profile_settings["enabled"]
                else presence.anchor,
                prepared.model_conditions.get("height"),
                prepared.model_conditions.get("profile"),
                selection.geometry,
            )
        self.scaler.update()
        if self.structure_every_steps and (step + 1) % self.structure_every_steps == 0:
            self.diagnostics.update(
                structure_metrics(clean_probs, real.images, self.critic_groups)
            )
        self.completed_steps = step + 1
        self.diagnostics["amp/scale"] = self.scaler.get_scale()
        return self.finish_step(
            prepared=prepared,
            denoiser_update=denoiser_update,
            critic_vals=critic_vals,
            r1=r1,
            critic_global=critic_global,
            critic_local=critic_local,
            critic_connectivity=critic_connectivity,
            connectivity_r1=connectivity_r1,
        )

    def _sample_fake_pairs(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        prepared: StepPreparation,
    ) -> SampledPairs:
        real = prepared.real
        selection = prepared.selection
        anchor = None if selection is None else selection.condition
        measured = None if selection is None else selection.measured
        anchor_present = prepared.presence.anchor
        visible = (
            None if anchor is None else self.visible_anchor(anchor, anchor_present)
        )
        excluded = (
            None if measured is None else self.visible_anchor(measured, anchor_present)
        )
        result = SampledPairs(pairs={})
        volume_height = prepared.model_conditions.get("height")
        profile = (
            prepared.model_conditions.get("profile")
            if self.profile_settings.get("critic_enabled", False)
            else None
        )
        profile_present = prepared.presence.vf.to(self.device)
        for axis in self.active_axes:
            height = volume_height if axis in real.heights else None
            if height is not None and profile is not None:
                field = profile_field(
                    profile * 2 - 1, previous.shape[-3:]
                ) * profile_present.reshape(-1, 1, 1, 1, 1)
                height = torch.cat((height, field), 1)
            pairs = sample_pairs(
                previous,
                current,
                axis,
                self.slice_pairs_per_axis,
                tuple(real.images[axis].shape[-2:]),
                anchor=visible,
                measured=excluded,
                height=height,
            )
            pairs = self.critic_augment.apply_together(pairs, plane=axis)
            result.pairs[axis] = pairs[:2]
            if len(pairs) == 3:
                result.heights[axis] = pairs[2][:, :1]
                if profile is not None:
                    result.profiles[axis] = pairs[2][:, 1:]
        return result

    def _make_denoiser_batch(
        self,
        prepared: StepPreparation,
        connectivity_domains: torch.Tensor,
        fake: SampledPairs,
        connectivity_real: TripletBatch,
        connectivity_fake: TripletBatch,
        logits: torch.Tensor,
        clean_probs: torch.Tensor,
    ) -> DenoiserBatch:
        selection = prepared.selection
        return DenoiserBatch(
            transition=prepared.transition,
            connectivity_domains=connectivity_domains,
            critic_domains=prepared.critic_domains,
            fake=fake.pairs,
            connectivity_real=connectivity_real,
            connectivity_fake=connectivity_fake,
            logits=logits,
            clean_probs=clean_probs,
            anchor=None if selection is None else selection.condition,
            anchor_observed_mask=(
                None if selection is None else selection.observed_mask
            ),
            anchor_observed_axis_masks=(
                None if selection is None else selection.observed_axis_masks
            ),
            anchor_present=prepared.presence.anchor,
            anchor_ramp=prepared.anchor_ramp,
            connectivity_ramp=prepared.connectivity_ramp,
            target_vf=prepared.target_vf,
            vf_present=prepared.presence.vf,
            fake_heights=fake.heights,
            fake_profiles=fake.profiles,
            profile=prepared.model_conditions.get("profile"),
            coarse_target=prepared.coarse_target,
        )

    def prepare_step(
        self,
        step: int,
        transition: int | None,
    ) -> StepPreparation:
        self.denoiser.train()
        self.critics.train()
        if self.connectivity_critic is not None:
            self.connectivity_critic.train()

        domain = self.sample_target_domain()
        if self.critic_groups_by_domain is not None:
            self.critic_groups = self.critic_groups_by_domain[domain]
            self.axis_critics = {
                axis: group
                for group, axes in self.critic_groups.items()
                for axis in axes
            }
            self.active_axes = tuple(sorted(self.axis_critics))
        model_domain = self.sample_domain_condition(domain)
        batch_domains = self.select_batch_domains(domain)
        batches = self.get_batches(domain, batch_domains)
        own_batches = {axis: batches.images[axis] for axis in self.streams[domain]}
        critic_domains = self.make_critic_domains(
            domain,
            model_domain,
            batch_domains,
        )
        if self.bank is not None:
            indices = torch.randint(len(self.bank[domain]), (self.volume_batch_size,))
            low = self.bank[domain][indices].to(self.device)
            coarse, level = corrupt_coarse(
                low, self.coarse_corruption_probability, self.coarse_corruption_strength
            )
            conditions = {
                "domain": self.make_domain(model_domain, self.volume_batch_size),
                "coarse": resize_phases(coarse, (self.patch_size,) * 3),
                "corruption_level": level,
            }
            if self.height_data is not None:
                conditions["height"] = self.volume_height(
                    self.bank_origins[domain][indices],
                    domain,
                    None
                    if self.bank_extents is None
                    else self.bank_extents[domain][indices],
                )
            self.diagnostics["coarse_corruption_mse"] = (coarse - low).square().mean()
            self.diagnostics["coarse_corruption_level"] = level.mean()
            absent = torch.zeros(self.volume_batch_size, dtype=torch.bool)
            return StepPreparation(
                transition=self.sample_transition(False)
                if transition is None
                else transition,
                domain=domain,
                critic_domains=critic_domains,
                real=batches,
                selection=None,
                target_vf=low.new_zeros((self.volume_batch_size, self.num_phases)),
                presence=ConditionPresence(absent, absent),
                model_conditions=conditions,
                anchor_ramp=0.0,
                connectivity_ramp=0.0,
                coarse_target=low,
            )
        target_vf = compute_vf(own_batches, self.num_phases)
        target_vf = target_vf.unsqueeze(0).expand(self.volume_batch_size, -1)
        ramp = self.get_anchor_ramp(step)
        selection = (
            None
            if ramp == 0.0
            else self.sample_anchor(
                batches,
                self.patch_size,
                owned_axes=tuple(own_batches),
                domain=domain,
            )
        )
        anchor = None if selection is None else selection.condition
        presence = self.sample_condition_presence(selection is not None)
        model_conditions = self.make_model_conditions(
            anchor,
            target_vf,
            presence,
            self.make_domain(model_domain, self.volume_batch_size),
        )
        if self.height_data is not None:
            height = None if selection is None else selection.height
            if height is None:
                axis = next(a for a in self.streams[domain] if a in batches.origins)
                height = self.volume_height(
                    batches.origins[axis][: self.volume_batch_size],
                    domain,
                    batches.extents[axis][: self.volume_batch_size],
                )
            model_conditions["height"] = height
        if self.profile_settings["enabled"]:
            profile = None if selection is None else selection.profile
            if profile is None:
                axis = next(a for a in own_batches if a in batches.profiles)
                profile = batches.profiles[axis][: self.volume_batch_size]
            target_vf = profile.mean(-1)
            model_conditions.update(
                profile=profile,
                profile_present=presence.vf.to(self.device),
                vf=target_vf,
            )
        if transition is None:
            transition = self.sample_transition(anchor is not None)
        return StepPreparation(
            transition=transition,
            domain=domain,
            critic_domains=critic_domains,
            real=batches,
            selection=selection,
            target_vf=target_vf,
            presence=presence,
            model_conditions=model_conditions,
            anchor_ramp=ramp,
            connectivity_ramp=self.schedule_ramp(
                step, self.connectivity_start_step, self.connectivity_ramp_steps
            ),
        )

    def finish_step(
        self,
        prepared: StepPreparation,
        denoiser_update: DenoiserUpdate,
        critic_vals: list[float],
        r1: float,
        critic_global: float,
        critic_local: float,
        critic_connectivity: float,
        connectivity_r1: float,
    ) -> Metrics:
        if self.generator_updated:
            update_ema(self.ema_denoiser, self.denoiser, self.ema_decay)
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
                connectivity_ramp=prepared.connectivity_ramp,
                anchor_neighbor_agreement=self.diagnostics.get(
                    "anchor/neighbor_agreement"
                ),
                anchor_neighbor_excess_jump=self.diagnostics.get(
                    "anchor/neighbor_excess_jump"
                ),
                generator_global=denoiser_update.global_loss,
                generator_local=denoiser_update.local_loss,
                critic_global=critic_global,
                critic_local=critic_local,
                vf_loss=denoiser_update.vf,
                vf_active=presence.vf.any(),
                anchor_input_active_fraction=presence.anchor.to(torch.float32).mean(),
                vf_active_fraction=presence.vf.to(torch.float32).mean(),
                normal_transition_loss=denoiser_update.normal_transition,
                anchor_coarse_loss=denoiser_update.anchor_coarse,
                anchor_pixel_loss=denoiser_update.anchor_pixel,
                anchor_shared=(
                    prepared.selection is not None
                    and prepared.selection.source == "shared"
                    and presence.anchor.any()
                ),
                diagnostics=dict(self.diagnostics),
            )
        )
        if metrics.diagnostics.get("anchor/boundary_pairs", 0) == 0:
            for key in ("anchor/neighbor_agreement", "anchor/neighbor_excess_jump"):
                if key in metrics.diagnostics:
                    metrics.diagnostics[key] = None
            metrics = replace(
                metrics,
                anchor_neighbor_agreement=None,
                anchor_neighbor_excess_jump=None,
            )
        return metrics

    def make_connectivity_triplets(
        self,
        prediction: torch.Tensor,
        reference_prediction: torch.Tensor | None,
        anchor: AnchorCondition | None,
        transition: int,
        visible: torch.Tensor | None = None,
        source: str | None = None,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
        profile_present: torch.Tensor | None = None,
    ) -> tuple[TripletBatch, TripletBatch]:
        empty = TripletBatch(
            values=prediction.new_empty(
                (0, 3, self.num_phases, prediction.shape[-2], prediction.shape[-1])
            ),
            axes=torch.empty(0, device=self.device, dtype=torch.long),
            gaps=torch.empty(0, device=self.device, dtype=torch.long),
            center_slots=torch.empty(0, device=self.device, dtype=torch.long),
        )
        enabled = self.connectivity_weight > 0.0 or self.normal_transition_weight > 0.0
        if (
            not enabled
            or transition != 0
            or anchor is None
            or source != "multi"
            or reference_prediction is None
        ):
            return empty, empty
        anchor = self.visible_anchor(anchor, visible)

        if profile is not None:
            field = profile_field(
                profile * 2 - 1, prediction.shape[-3:]
            ) * profile_present.reshape(-1, 1, 1, 1, 1)
            height = torch.cat((height, field), 1)
        real, fake = self.anchor_triplets.sample(
            prediction,
            reference_prediction,
            anchor,
            height=height,
        )
        augmented = self.critic_augment.apply_together(
            (real.values, fake.values)
            if real.height is None
            else (real.values, fake.values, real.height),
            plane=real.axes,
        )
        return (
            TripletBatch(
                values=augmented[0],
                height=None if real.height is None else augmented[2][:, :, :1],
                profile=augmented[2][:, :, 1:] if profile is not None else None,
                axes=real.axes,
                gaps=real.gaps,
                center_slots=real.center_slots,
            ),
            TripletBatch(
                values=augmented[1],
                height=None if real.height is None else augmented[2][:, :, :1],
                profile=augmented[2][:, :, 1:] if profile is not None else None,
                axes=fake.axes,
                gaps=fake.gaps,
                center_slots=fake.center_slots,
            ),
        )

    @staticmethod
    def get_connectivity_domains(
        critic_domains: dict[int, int],
        triplets: TripletBatch,
    ) -> torch.Tensor:
        return torch.tensor(
            [critic_domains.get(axis, NULL_DOMAIN) for axis in AXES],
            device=triplets.values.device,
            dtype=torch.long,
        )[triplets.axes]

    @staticmethod
    def visible_anchor(
        anchor: AnchorCondition,
        visible: torch.Tensor | None,
    ) -> AnchorCondition:
        if visible is None:
            return anchor
        mask = visible.to(anchor.mask.device).reshape(-1, 1, 1, 1, 1)
        return replace(
            anchor,
            image=anchor.image * mask,
            mask=anchor.mask & mask,
            axis_masks=anchor.axis_masks & mask,
            active_batches=tuple(visible.nonzero().flatten().tolist()),
        )

    def sample_transition(self, anchored: bool) -> int:
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
        return int(torch.randint(len(self.streams), ()).item())

    def get_batches(
        self,
        domain: int,
        batch_domains: dict[int, int] | None = None,
    ) -> RealBatch:
        if batch_domains is None:
            batch_domains = self.select_batch_domains(domain)
        batches = RealBatch(images={}, domains=batch_domains)
        for axis in self.active_axes:
            batch = self.streams[batch_domains[axis]][axis].next()
            images = batch["image"]
            if self.height_data is not None:
                origins = batch["height_origin"]
                if axis != 0:
                    batches.origins[axis] = origins
                    batches.extents[axis] = batch["height_extent"]
                    batches.geometry[axis] = [
                        {
                            "image_id": image_id,
                            "height_origin": float(origin),
                            "height_extent": float(extent),
                            "crop_origin": crop.tolist(),
                            "source_shape": shape.tolist(),
                        }
                        for image_id, origin, extent, crop, shape in zip(
                            batch["image_id"],
                            origins,
                            batch["height_extent"],
                            batch["crop_origin"],
                            batch["source_shape"],
                            strict=True,
                        )
                    ]
                    batches.heights[axis] = height_field(
                        images.shape[-2:],
                        0,
                        origins,
                        self.height_data["crop_size"] / images.shape[-1],
                        batch["height_extent"],
                        self.device,
                    )
            batches.images[axis] = images.to(self.device, non_blocking=True)
            if self.profile_settings["enabled"] and axis in batches.origins:
                batches.profiles[axis] = rebin_profile(
                    image_profile(
                        batches.images[axis], bins=self.profile_settings["num_bins"]
                    ),
                    self.patch_size,
                )
        return batches

    def volume_height(self, origins, domain, extents=None):
        data = self.height_data
        return height_field(
            (self.patch_size,) * 3,
            0,
            origins,
            data["crop_size"] / self.patch_size,
            data["height_extents"][domain] if extents is None else extents,
            self.device,
        )

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

    def make_domain(self, domain: int, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            domain,
            device=self.device,
            dtype=torch.long,
        )

    def sample_condition_presence(self, has_anchor: bool) -> ConditionPresence:
        batch = self.volume_batch_size
        anchor = torch.full(
            (batch,),
            has_anchor,
            device="cpu",
            dtype=torch.bool,
        )
        vf = torch.ones(batch, device="cpu", dtype=torch.bool)
        random = torch.rand(batch, device="cpu")
        if has_anchor:
            probability = self.cfg_drop_each_probability
            joint_null = random < probability
            anchor_null = (random >= probability) & (random < 2.0 * probability)
            vf_null = (random >= 2.0 * probability) & (random < 3.0 * probability)
            anchor = ~(joint_null | anchor_null)
            vf = ~(joint_null | vf_null)
        else:
            vf = random >= 2.0 * self.cfg_drop_each_probability
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
            "vf_present": presence.vf.to(target_vf.device),
        }
        if anchor is not None:
            visible = presence.anchor.to(anchor.mask.device).reshape(-1, 1, 1, 1, 1)
            conditions["anchor_image"] = anchor.image
            conditions["anchor_mask"] = anchor.mask * visible.to(anchor.mask.dtype)
        return conditions

    def get_anchor_ramp(self, step: int) -> float:
        return self.schedule_ramp(step, self.anchor_start_step, self.anchor_ramp_steps)

    @staticmethod
    def schedule_ramp(step: int, start: int, ramp: int) -> float:
        if step < start:
            return 0.0
        if ramp == 0:
            return 1.0
        return min((step - start + 1) / ramp, 1.0)

    def sample_anchor(
        self,
        batches: RealBatch,
        volume_size: int,
        owned_axes: tuple[int, ...] | None = None,
        domain: int = 0,
    ) -> AnchorSelection | None:
        probability = self.anchor_training_probability
        if probability <= 0.0:
            return None
        if probability < 1.0 and not bool(torch.rand(()) < probability):
            return None
        if self.use_multi_anchor_next:
            replay = self.anchor_bank.sample(
                domain, self.volume_batch_size, self.device
            )
            self.use_multi_anchor_next = False
            if replay is not None:
                return AnchorSelection(
                    condition=replay.condition,
                    observed_mask=replay.measured.mask,
                    observed_axis_masks=replay.measured.axis_masks,
                    source="multi",
                    measured=replay.measured,
                    reference=replay.reference,
                    height=replay.height,
                    profile=replay.profile,
                    geometry=replay.geometry,
                )
        self.use_multi_anchor_next = True
        return self.sample_real_anchor(
            batches,
            volume_size,
            owned_axes=owned_axes,
        )

    def sample_real_anchor(
        self,
        batches: RealBatch,
        volume_size: int,
        owned_axes: tuple[int, ...] | None = None,
    ) -> AnchorSelection:
        if owned_axes is None:
            owned_axes = tuple(batches.images)
        shared_axes = tuple(axis for axis in batches.images if axis not in owned_axes)
        use_shared = (
            bool(shared_axes)
            and self.anchor_shared_axis_probability > 0.0
            and bool(torch.rand(()) < self.anchor_shared_axis_probability)
        )
        axes = shared_axes if use_shared else owned_axes
        if batches.origins:
            # An xy section has no observed absolute height along z.
            axes = tuple(a for a in axes if a in batches.origins)
            if not axes:
                axes = tuple(a for a in owned_axes if a in batches.origins)
                use_shared = False
        axis = axes[int(torch.randint(len(axes), ()).item())]
        images = batches.images[axis]
        batch_indices = torch.randperm(
            images.shape[0],
            device="cpu",
        )[: self.volume_batch_size]
        selected = images.index_select(0, batch_indices.to(images.device))
        shape = tuple(min(volume_size, size) for size in selected.shape[-2:])
        selected = crop_images(selected, shape)
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
        condition = encode_anchors(
            (plane,),
            batch_size=self.volume_batch_size,
            num_phases=self.num_phases,
            volume_size=volume_size,
            device=self.device,
            dtype=torch.float32,
            reconcile=False,
            validate=False,
        )
        return AnchorSelection(
            condition=condition,
            observed_mask=condition.mask,
            observed_axis_masks=condition.axis_masks,
            source="shared" if use_shared else "real",
            measured=condition,
            height=self.volume_height(
                batches.origins[axis][batch_indices],
                batches.domains[axis],
                batches.extents[axis][batch_indices],
            )
            if axis in batches.origins
            else None,
            profile=batches.profiles[axis][batch_indices.to(self.device)]
            if axis in batches.profiles
            else None,
            geometry=[batches.geometry[axis][int(index)] for index in batch_indices]
            if axis in batches.origins
            else None,
        )

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
            logits = self.denoiser.compute_logits(
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

    def update_critics(
        self,
        transition: int,
        fake: SampledPairs,
        batches: RealBatch,
        step: int,
        domains: dict[int, int],
        profile_present: torch.Tensor | None = None,
    ) -> tuple[list[float], float, float, float]:
        critic_losses = [0.0] * len(AXES)
        r1_sum = 0.0
        global_sum = 0.0
        local_sum = 0.0
        local_weight = self.critic_local_weight
        groups = self.active_groups(fake.pairs)
        for group, axes in groups.items():
            count = self.updates[group]
            regularize = (count + 1) % self.r1_interval == 0
            apply_r1 = self.r1_gamma > 0.0 and regularize
            apply_r2 = self.r2_gamma > 0.0 and regularize
            weight = 1.0 / (len(axes) * len(groups))
            critic = self.critics[group]
            optimizer = self.critic_optims[group]
            optimizer.zero_grad(set_to_none=True)
            for axis in axes:
                images = batches.images[axis]
                if (
                    images.ndim != 4
                    or images.shape[1] != self.num_phases
                    or not images.is_floating_point()
                ):
                    raise ValueError(
                        "training images must be floating-point [B,C,H,W] phase fractions."
                    )
                real = images.mul(2.0).sub(1.0)
                real_time = self.make_time(transition, real.shape[0])
                real_prev, real_curr = self.diffusion.sample_pair(
                    real,
                    transition,
                )
                real_height = batches.heights.get(axis)
                if real_height is not None and axis in fake.profiles:
                    profile = (batches.profiles[axis] * 2 - 1)[..., None].expand(
                        -1, -1, *real_height.shape[-2:]
                    )
                    present = profile_present[
                        torch.randint(
                            len(profile_present),
                            (len(profile),),
                            device=self.device,
                        )
                    ]
                    real_height = torch.cat(
                        (real_height, profile * present.reshape(-1, 1, 1, 1)), 1
                    )
                inputs = (
                    (real_prev, real_curr)
                    if real_height is None
                    else (real_prev, real_curr, real_height)
                )
                augmented = self.critic_augment.apply_together(inputs, plane=axis)
                real_prev, real_curr = augmented[:2]
                real_conditions = (
                    {} if real_height is None else {"height": augmented[2][:, :1]}
                )
                fake_conditions = (
                    {} if axis not in fake.heights else {"height": fake.heights[axis]}
                )

                if real_height is not None and axis in fake.profiles:
                    real_conditions["profile"] = augmented[2][:, 1:]
                    fake_conditions["profile"] = fake.profiles[axis]

                real_prev.requires_grad_(regularize)
                real_curr.requires_grad_(regularize)
                fake_prev, fake_curr = fake.pairs[axis]
                fake_prev = fake_prev.detach().float().requires_grad_(apply_r2)
                fake_curr = fake_curr.detach().float()
                fake_time = self.make_time(transition, fake_prev.shape[0])
                real_domain = self.make_domain(domains[axis], real_prev.shape[0])
                fake_domain = self.make_domain(domains[axis], fake_prev.shape[0])

                autocast = self.autocast(self.amp_enabled and not regularize)
                with autocast:
                    real_score = critic(
                        real_prev, real_curr, real_time, real_domain, **real_conditions
                    )
                    fake_score = critic(
                        fake_prev, fake_curr, fake_time, fake_domain, **fake_conditions
                    )
                    losses = get_critic_loss(real_score, fake_score)
                    loss = losses.combine(local_weight)
                global_sum += losses.global_loss.detach() * weight
                local_sum += losses.local_loss.detach() * weight
                self.diagnostics[f"score/{group}/{axis}/real"] = (
                    real_score.logits_global.detach().mean()
                )
                self.diagnostics[f"score/{group}/{axis}/fake"] = (
                    fake_score.logits_global.detach().mean()
                )
                if regularize:
                    norms = input_gradient_norms(
                        real_score.logits_global.sum(),
                        (real_prev, real_curr),
                    )
                    for name, norm in zip(("previous", "current"), norms):
                        self.diagnostics[f"input_gradient/{group}/{axis}/{name}"] = norm
                if apply_r1:
                    r1 = get_critic_r1(
                        real_score,
                        (real_prev,),
                    )
                    penalty = r1.combine(local_weight)
                    r1_sum += penalty.detach() * weight
                    loss = loss + 0.5 * self.r1_gamma * self.r1_interval * penalty
                if apply_r2:
                    penalty = get_critic_r1(fake_score, (fake_prev,)).combine(
                        local_weight
                    )
                    self.diagnostics[f"r2/{group}/{axis}"] = penalty.detach()
                    loss = loss + 0.5 * self.r2_gamma * self.r1_interval * penalty
                loss = validate_loss(loss, f"critic {group}/{axis}")
                self.scaler.scale(loss * weight).backward()
                critic_losses[axis] = loss.detach() * weight
            if step_optimizer(optimizer, self.scaler, self.diagnostics, group):
                self.updates[group] += 1
        return critic_losses, r1_sum, global_sum, local_sum

    def active_groups(self, fake):
        groups = {
            group: tuple(axis for axis in axes if len(fake[axis][0]))
            for group, axes in self.critic_groups.items()
        }
        return {group: axes for group, axes in groups.items() if axes}

    def update_connectivity_critic(
        self,
        real: torch.Tensor,
        fake: TripletBatch,
        step: int,
        domains: torch.Tensor,
    ) -> tuple[float, float]:
        if not len(fake):
            return 0.0, 0.0

        count = self.updates["connectivity"]
        regularize = (count + 1) % self.r1_interval == 0
        apply_r1 = self.r1_gamma > 0.0 and regularize
        apply_r2 = self.r2_gamma > 0.0 and regularize
        self.connectivity_optim.zero_grad(set_to_none=True)
        real = real.detach().float().requires_grad_(apply_r1)
        fake_values = fake.values.detach().float().requires_grad_(apply_r2)
        autocast = self.autocast(self.amp_enabled and not regularize)
        with autocast:
            real_score = self.connectivity_critic(
                real,
                fake.axes,
                fake.gaps,
                domains,
                **({"height": fake.height} if fake.height is not None else {}),
                **({"profile": fake.profile} if fake.profile is not None else {}),
            )
            fake_score = self.connectivity_critic(
                fake_values,
                fake.axes,
                fake.gaps,
                domains,
                **({"height": fake.height} if fake.height is not None else {}),
                **({"profile": fake.profile} if fake.profile is not None else {}),
            )
            losses = get_critic_loss(
                real_score,
                fake_score,
            )
            loss = losses.combine(self.critic_local_weight)
        adversarial = loss.detach()
        r1_value = 0.0
        if apply_r1:
            r1 = get_critic_r1(real_score, (real,))
            penalty = r1.combine(self.critic_local_weight)
            r1_value = penalty.detach()
            loss = loss + 0.5 * self.r1_gamma * self.r1_interval * penalty
        if apply_r2:
            penalty = get_critic_r1(fake_score, (fake_values,)).combine(
                self.critic_local_weight
            )
            self.diagnostics["r2/connectivity"] = penalty.detach()
            loss = loss + 0.5 * self.r2_gamma * self.r1_interval * penalty
        self.diagnostics["regularization/connectivity"] = int(regularize)
        loss = validate_loss(loss, "connectivity")
        self.scaler.scale(loss).backward()
        if step_optimizer(
            self.connectivity_optim, self.scaler, self.diagnostics, "connectivity"
        ):
            self.updates["connectivity"] += 1
        return adversarial, r1_value

    def update_denoiser(
        self,
        batch: DenoiserBatch,
    ) -> DenoiserUpdate:
        self.denoiser_optim.zero_grad(set_to_none=True)
        for critic in self.critics.values():
            critic.requires_grad_(False)
        if self.connectivity_critic is not None:
            self.connectivity_critic.requires_grad_(False)
        try:
            local_weight = self.critic_local_weight
            with self.autocast():
                head = self._adversarial_loss(batch)
                global_loss, local_loss = head.global_loss, head.local_loss
                adversarial_loss = global_loss + local_weight * local_loss
                connectivity_loss = adversarial_loss.new_zeros(())
                if len(batch.connectivity_fake) and self.connectivity_weight > 0.0:
                    connectivity_scores = self.connectivity_critic(
                        batch.connectivity_fake.values,
                        batch.connectivity_fake.axes,
                        batch.connectivity_fake.gaps,
                        batch.connectivity_domains,
                        **(
                            {"height": batch.connectivity_fake.height}
                            if batch.connectivity_fake.height is not None
                            else {}
                        ),
                        **(
                            {"profile": batch.connectivity_fake.profile}
                            if batch.connectivity_fake.profile is not None
                            else {}
                        ),
                    )
                    connectivity_head = get_generator_loss(
                        connectivity_scores,
                    )
                    connectivity_loss = connectivity_head.combine(local_weight)
                normal_loss = adversarial_loss.new_zeros(())
                if len(batch.connectivity_fake) and self.normal_transition_weight > 0.0:
                    normal_loss = compute_transition_loss(
                        batch.connectivity_real,
                        batch.connectivity_fake,
                    )
                anchor_loss = adversarial_loss.new_zeros(())
                anchor_coarse = adversarial_loss.new_zeros(())
                anchor_pixel = adversarial_loss.new_zeros(())
                anchor_accuracy = adversarial_loss.new_zeros(())
                if batch.anchor is not None:
                    anchor_result = self.anchor_loss(
                        batch.logits,
                        batch.anchor,
                        batch.anchor_present,
                        batch.anchor_observed_mask,
                        batch.anchor_observed_axis_masks,
                    )
                    anchor_loss = anchor_result.total
                    anchor_coarse = anchor_result.coarse
                    anchor_pixel = anchor_result.pixel
                    anchor_accuracy = anchor_result.accuracy
                vf_loss = compute_vf_loss(
                    batch.clean_probs,
                    batch.target_vf,
                    batch.vf_present,
                )
                total = (
                    adversarial_loss
                    + batch.anchor_ramp * anchor_loss
                    + batch.connectivity_ramp
                    * (
                        self.connectivity_weight * connectivity_loss
                        + self.normal_transition_weight * normal_loss
                    )
                    + self.vf_loss_weight * vf_loss
                )
                if batch.profile is not None:
                    total = total + self._profile_loss(batch)
                if batch.coarse_target is not None:
                    total = total + self.consistency_weight * self._consistency_loss(
                        batch
                    )
            total = validate_loss(total, "generator")
            self.scaler.scale(total).backward()
            self.generator_updated = step_optimizer(
                self.denoiser_optim, self.scaler, self.diagnostics, "generator"
            )
            if self.generator_updated:
                self.updates["generator"] += 1
        finally:
            for critic in self.critics.values():
                critic.requires_grad_(True)
            if self.connectivity_critic is not None:
                self.connectivity_critic.requires_grad_(True)
        return DenoiserUpdate(
            adversarial=adversarial_loss.detach(),
            total=total.detach(),
            global_loss=global_loss.detach(),
            local_loss=local_loss.detach(),
            connectivity=connectivity_loss.detach(),
            normal_transition=normal_loss.detach(),
            anchor=anchor_loss.detach(),
            anchor_coarse=anchor_coarse.detach(),
            anchor_pixel=anchor_pixel.detach(),
            anchor_accuracy=anchor_accuracy.detach(),
            vf=vf_loss.detach(),
        )

    def _adversarial_loss(self, batch: DenoiserBatch) -> HeadLoss:
        heads = []
        local_weight = self.critic_local_weight
        groups = self.active_groups(batch.fake)
        diagnose = (self.updates["generator"] + 1) % self.r1_interval == 0
        for axis in self.active_axes:
            fake_prev, fake_curr = batch.fake[axis]
            if not len(fake_prev):
                continue
            if diagnose:
                # A separate leaf measures conditional sensitivity without
                # reconnecting the preceding reverse chain to the generator.
                fake_curr = fake_curr.detach().requires_grad_(True)
            time = self.make_time(batch.transition, fake_prev.shape[0])
            domains = self.make_domain(
                batch.critic_domains[axis],
                fake_prev.shape[0],
            )
            conditions = {}
            if axis in batch.fake_heights:
                conditions["height"] = batch.fake_heights[axis]
                if axis in batch.fake_profiles:
                    conditions["profile"] = batch.fake_profiles[axis]
            scores = self.critics[self.axis_critics[axis]](
                fake_prev,
                fake_curr,
                time,
                domains,
                **conditions,
            )
            head = get_generator_loss(scores)
            if diagnose:
                # Undo only the batch mean, retaining local-head weights
                # and pyramid averaging from the actual generator loss.
                norms = input_gradient_norms(
                    head.combine(local_weight) * len(fake_prev),
                    (fake_prev, fake_curr),
                )
                prefix = (
                    f"generator_input_gradient/{self.axis_critics[axis]}"
                    f"/{axis}/t{batch.transition}"
                )
                for name, norm in zip(("previous", "current"), norms):
                    self.diagnostics[f"{prefix}/{name}"] = norm
            count = len(groups[self.axis_critics[axis]]) * len(groups)
            heads.append(HeadLoss(head.global_loss / count, head.local_loss / count))
        global_loss = (
            torch.stack([loss.global_loss for loss in heads]).sum()
            if heads
            else batch.logits.sum() * 0
        )
        local_loss = (
            torch.stack([loss.local_loss for loss in heads]).sum()
            if heads
            else batch.logits.sum() * 0
        )
        return HeadLoss(global_loss, local_loss)

    def _profile_loss(self, batch: DenoiserBatch) -> torch.Tensor:
        profile_loss = compute_profile_loss(
            batch.clean_probs,
            batch.profile,
            batch.vf_present,
            self.profile_settings["num_bins"],
            self.profile_gradient_weight,
            self.profile_weight,
        )
        self.diagnostics["profile/loss"] = profile_loss.detach()
        soft = batch.clean_probs.detach().mean((-1, -2))
        labels = phase_profile(batch.clean_probs.detach().argmax(1), self.num_phases)
        active = batch.vf_present.to(soft).reshape(-1, 1, 1)
        denom = active.sum().clamp_min(1) * soft.shape[1] * soft.shape[2]
        self.diagnostics["profile/soft_mae"] = (
            (soft - batch.profile).abs() * active
        ).sum() / denom
        self.diagnostics["profile/label_mae"] = (
            (labels - batch.profile).abs() * active
        ).sum() / denom
        return profile_loss

    def _consistency_loss(self, batch: DenoiserBatch) -> torch.Tensor:
        consistency, error = consistency_loss(
            batch.clean_probs,
            batch.coarse_target,
            self.consistency_tolerance,
        )
        self.diagnostics["consistency"] = consistency.detach()
        self.diagnostics["lr_mse"] = error.detach()
        depth = batch.coarse_target.shape[-3]
        target_profile = phase_profile(batch.coarse_target.detach(), self.num_phases)
        soft_profile = phase_profile(batch.clean_probs.detach(), self.num_phases, depth)
        label_profile = phase_profile(
            batch.clean_probs.detach().argmax(1), self.num_phases, depth
        )
        self.diagnostics["profile/sr_coarse_soft_mae"] = (
            (soft_profile - target_profile).abs().mean()
        )
        self.diagnostics["profile/sr_coarse_label_mae"] = (
            (label_profile - target_profile).abs().mean()
        )
        return consistency

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
