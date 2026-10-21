import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from src.anchor import AnchorCondition
from src.data.augment import CriticAugment
from src.data.loader import BatchStream
from src.data.slice import (
    AnchorTripletSampler,
    BandBatch,
    TripletBatch,
    sample_pairs,
    sample_seam_bands,
)
from src.evaluate.anchor import anchor_boundary_metrics
from src.evaluate.structure import structure_metrics
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.model.layers import NULL_DOMAIN
from src.plane import AXES, PLANE_AXES
from src.prepare.profile import profile_field
from src.train.batch import (
    DenoiserBatch,
    DenoiserUpdate,
    RealBatch,
    SampledPairs,
    StepPreparation,
)
from src.train.conditioning import ConditionSampler
from src.train.ema import update_ema
from src.train.loss.anchor import SoftAnchorLoss
from src.train.loss.anchor_continuity import compute_anchor_continuity_loss
from src.train.loss.anchor_neighborhood import compute_anchor_neighborhood_loss
from src.train.loss.denoiser import DenoiserLossSettings, denoiser_objective
from src.train.loss.gan import (
    active_groups,
    critic_objective,
    score_plane,
    score_triplets,
)
from src.train.loss.transition import (
    compute_real_transition_loss,
    transition_statistics,
)
from src.train.metrics import Metrics, collect_metrics
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
    seam_critic: nn.Module | None = None
    seam_optim: torch.optim.Optimizer | None = None
    critic_augment: CriticAugment | None = None
    coarse_bank: dict | None = None
    bank_origins: dict | None = None
    bank_extents: dict | None = None
    critic_groups_by_domain: dict | None = None
    data_fingerprint: dict[str, str] = field(default_factory=dict)
    cfg: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerSettings:
    volume_batch_size: int
    num_phases: int
    patch_size: int
    slice_pairs_per_axis: int
    ema_decay: float
    r1_gamma: float
    r1_interval: int
    loss: DenoiserLossSettings
    anchor_training_probability: float
    anchor_start_step: int
    anchor_ramp_steps: int
    cfg_drop_each_probability: float
    latent_channels: int
    amp_enabled: bool
    domain_dropout: float
    anchor_pixel_loss_weight: float
    anchor_shared_axis_probability: float
    connectivity_max_gap: int = 1
    group_statistics_start_step: int = 0
    group_statistics_ramp_steps: int = 5000
    r2_gamma: float = 0.0
    connectivity_start_step: int = 0
    connectivity_ramp_steps: int = 20000
    connectivity_windows_per_plane: int = 4
    anchor_bank_capacity: int = 4
    anchor_plane_spacing: int = 16
    structure_every_steps: int = 100
    coarse_corruption_probability: float = 0.0
    coarse_corruption_strength: float = 0.0
    height_data: dict | None = None
    volume_permute_z: bool = True
    volume_flip_z: bool = True
    profile_settings: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "num_bins": 16,
            "critic_enabled": False,
        }
    )


class Trainer:
    def __init__(
        self,
        components: TrainerComponents,
        settings: TrainerSettings,
    ) -> None:
        self.settings = settings
        self.denoiser = components.denoiser
        self.ema_denoiser = components.ema_denoiser
        self.critics = components.critics
        self.connectivity_critic = components.connectivity_critic
        self.seam_critic = components.seam_critic
        self.streams = components.streams
        self.data_fingerprint = components.data_fingerprint
        self.cfg = components.cfg
        self.critic_groups_by_domain = components.critic_groups_by_domain
        self._use_critic_groups(
            {
                group: tuple(PLANE_AXES[plane] for plane in group.split("_"))
                for group in self.critics
            }
            if self.critic_groups_by_domain is None
            else self.critic_groups_by_domain[0]
        )
        self.conditions = ConditionSampler(
            self.settings,
            self.streams,
            components.device,
            components.diffusion.timesteps,
            self.active_axes,
            components.coarse_bank,
            components.bank_origins,
            components.bank_extents,
        )
        self.diffusion = components.diffusion
        self.denoiser_optim = components.denoiser_optim
        self.critic_optims = components.critic_optims
        self.connectivity_optim = components.connectivity_optim
        self.seam_optim = components.seam_optim
        self.scaler = components.scaler
        self.device = components.device
        pool_exponent = math.ceil(math.log2(self.denoiser.downsample_factor) / 2.0)
        self.anchor_loss = SoftAnchorLoss(
            pool_size=2**pool_exponent,
            pixel_weight=settings.anchor_pixel_loss_weight,
        )
        self.anchor_triplets = AnchorTripletSampler(
            max_gap=settings.connectivity_max_gap,
            windows_per_plane=settings.connectivity_windows_per_plane,
        )
        self.updates = {
            name: 0 for name in (*self.critics, "connectivity", "generator")
        }
        if self.seam_critic is not None:
            self.updates["seam"] = 0
        self.completed_steps = 0
        self.path_maps = []
        self.diagnostics = {}
        self.generator_updated = True
        self.critic_augment = (
            CriticAugment()
            if components.critic_augment is None
            else components.critic_augment
        )

    def _use_critic_groups(self, groups: dict[str, tuple[int, ...]]) -> None:
        self.critic_groups = groups
        self.axis_critics = {
            axis: group for group, axes in groups.items() for axis in axes
        }
        self.active_axes = tuple(sorted(self.axis_critics))

    def step(
        self,
        step: int,
        transition: int | None = None,
    ) -> Metrics:
        self.diagnostics = {}
        prepared = self.prepare_step(step, transition)
        volume_size = self.settings.patch_size
        transition = prepared.transition
        real = prepared.real
        critic_domains = prepared.critic_domains
        presence = prepared.presence
        selection = prepared.selection
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
        profile_present = presence.vf.to(self.device)
        profile = (
            prepared.model_conditions.get("profile")
            if self.settings.profile_settings["critic_enabled"]
            else None
        )
        has_profile = profile is not None
        critic_height = self._with_profile(
            prepared.model_conditions.get("height"),
            profile,
            profile_present,
            prediction.shape[-3:],
        )
        fake = self._sample_fake_pairs(
            previous, current, prepared, critic_height, has_profile
        )
        connectivity_real, connectivity_fake = self.make_connectivity_triplets(
            prediction,
            None if selection is None else selection.reference,
            anchor,
            transition,
            self._anchor_visibility(presence),
            None if selection is None else selection.source,
            height=critic_height,
            has_profile=has_profile,
        )
        if selection is not None and transition == 0:
            self.diagnostics.update(
                anchor_boundary_metrics(prediction, measured.visible(presence.anchor))
            )
        connectivity_domains = self.get_axis_domains(
            prepared.critic_domains,
            connectivity_fake.axes,
        )
        seam_real, seam_fake = self.make_seam_bands(prediction, anchor, presence.anchor)
        seam_domains = self.get_axis_domains(prepared.critic_domains, seam_fake.axes)

        critic_vals, r1 = self.update_critics(
            transition,
            fake,
            real,
            critic_domains,
            profile_present=profile_present,
        )
        # Disabled critics receive empty batches, which skip their updates.
        (
            critic_connectivity,
            connectivity_r1,
            critic_connectivity_global,
            critic_connectivity_local,
        ) = self.update_connectivity_critic(
            connectivity_real,
            connectivity_fake,
            connectivity_domains,
        )
        self.update_seam_critic(
            transition, seam_real, seam_fake, prepared.critic_domains
        )
        denoiser_update = self.update_denoiser(
            self._make_denoiser_batch(
                prepared=prepared,
                connectivity_domains=connectivity_domains,
                fake=fake,
                connectivity_fake=connectivity_fake,
                logits=logits,
                clean_probs=clean_probs,
                seam_fake=seam_fake,
                seam_domains=seam_domains,
            )
        )
        if (
            self.generator_updated
            and transition == 0
            and selection is not None
            and selection.source in ("real", "shared")
        ):
            self.conditions.anchor_bank.add(
                prepared.domain,
                prediction,
                measured,
                self._anchor_visibility(presence),
                prepared.model_conditions.get("height"),
                prepared.model_conditions.get("profile"),
                selection.geometry,
            )
        self.scaler.update()
        if (
            self.settings.structure_every_steps
            and (step + 1) % self.settings.structure_every_steps == 0
        ):
            self.diagnostics.update(
                structure_metrics(clean_probs, real.images, self.critic_groups)
            )
        self.completed_steps = step + 1
        self.diagnostics["amp/scale"] = self.scaler.get_scale()
        if self.generator_updated:
            update_ema(self.ema_denoiser, self.denoiser, self.settings.ema_decay)
        return collect_metrics(
            volume_size=self.settings.patch_size,
            diagnostics=self.diagnostics,
            prepared=prepared,
            denoiser_update=denoiser_update,
            critic_vals=critic_vals,
            r1=r1,
            critic_connectivity=critic_connectivity,
            critic_connectivity_global=critic_connectivity_global,
            critic_connectivity_local=critic_connectivity_local,
            connectivity_r1=connectivity_r1,
        )

    def prepare_step(
        self,
        step: int,
        transition: int | None,
    ) -> StepPreparation:
        self.denoiser.train()
        for critic in self.critic_modules():
            critic.train()

        domain = self.conditions.sample_target_domain()
        if self.critic_groups_by_domain is not None:
            self._use_critic_groups(self.critic_groups_by_domain[domain])
        self.conditions.active_axes = self.active_axes
        model_domain = self.conditions.sample_domain_condition(domain)
        batch_domains = self.conditions.select_batch_domains(domain)
        batches = self.conditions.get_batches(domain, batch_domains)
        critic_domains = self.conditions.make_critic_domains(
            domain,
            model_domain,
            batch_domains,
        )
        if self.conditions.bank is not None:
            prepared = self.conditions.prepare_sr_step(
                domain, model_domain, critic_domains, batches, transition
            )
        else:
            prepared = self.conditions.prepare_lr_step(
                step, domain, model_domain, critic_domains, batches, transition
            )
        self.diagnostics.update(prepared.diagnostics)
        ramp = self.conditions.schedule_ramp(
            step,
            self.settings.group_statistics_start_step,
            self.settings.group_statistics_ramp_steps,
        )
        return replace(prepared, group_statistics_ramp=ramp)

    def generate_pair(
        self,
        transition: int,
        model_conditions: dict[str, torch.Tensor],
        volume_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (
            self.settings.volume_batch_size,
            self.settings.num_phases,
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

    def _sample_fake_pairs(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        prepared: StepPreparation,
        volume_height: torch.Tensor | None,
        has_profile: bool,
    ) -> SampledPairs:
        real = prepared.real
        selection = prepared.selection
        anchor_present = prepared.presence.anchor
        visible = excluded = None
        if selection is not None:
            visible = selection.condition.visible(anchor_present)
            excluded = selection.measured.visible(anchor_present)
        result = SampledPairs(pairs={})
        for axis in self.active_axes:
            height = volume_height if axis in real.heights else None
            pairs = sample_pairs(
                previous,
                current,
                axis,
                self.settings.slice_pairs_per_axis,
                tuple(real.images[axis].shape[-2:]),
                anchor=visible,
                measured=excluded,
                height=height,
            )
            if not self.critic_augment.real_only:
                pairs = self.critic_augment.apply_together(pairs, plane=axis)
            result.pairs[axis] = pairs[:2]
            if len(pairs) == 3:
                result.heights[axis] = pairs[2][:, :1]
                if has_profile:
                    result.profiles[axis] = pairs[2][:, 1:]
        return result

    def _anchor_visibility(self, presence) -> torch.Tensor:
        # With profiles on, an anchor counts only where its profile is visible too.
        if self.settings.profile_settings["enabled"]:
            return presence.anchor & presence.vf
        return presence.anchor

    @staticmethod
    def _with_profile(height, profile, present, shape) -> torch.Tensor | None:
        """Append the masked profile field to the height channel for critics."""
        if height is None or profile is None:
            return height
        field = profile_field(profile * 2 - 1, shape) * present.reshape(-1, 1, 1, 1, 1)
        return torch.cat((height, field), 1)

    def _make_denoiser_batch(
        self,
        prepared: StepPreparation,
        connectivity_domains: torch.Tensor,
        fake: SampledPairs,
        connectivity_fake: TripletBatch,
        logits: torch.Tensor,
        clean_probs: torch.Tensor,
        seam_fake: BandBatch | None = None,
        seam_domains: torch.Tensor | None = None,
    ) -> DenoiserBatch:
        selection = prepared.selection
        # SR never selects anchors, so a selection implies an LR step.
        continuity = (
            selection is not None and self.settings.loss.anchor_continuity_weight > 0
        )
        real_transition = (
            self.settings.loss.real_transition_weight > 0 and prepared.transition == 0
        )
        statistics = (
            self._transition_statistics(prepared)
            if continuity or real_transition
            else None
        )
        return DenoiserBatch(
            transition=prepared.transition,
            connectivity_domains=connectivity_domains,
            critic_domains=prepared.critic_domains,
            fake=fake.pairs,
            connectivity_fake=connectivity_fake,
            logits=logits,
            clean_probs=clean_probs,
            anchor=None if selection is None else selection.condition,
            anchor_measured=None if selection is None else selection.measured,
            anchor_present=prepared.presence.anchor,
            anchor_ramp=prepared.anchor_ramp,
            connectivity_ramp=prepared.connectivity_ramp,
            target_vf=prepared.target_vf,
            vf_present=prepared.presence.vf,
            fake_heights=fake.heights,
            fake_profiles=fake.profiles,
            profile=prepared.model_conditions.get("profile"),
            coarse_target=prepared.coarse_target,
            coarse_mask=prepared.coarse_mask,
            real_transition_loss=(
                self._real_transition_loss(prepared, clean_probs, statistics)
                if real_transition
                else None
            ),
            anchor_continuity_loss=(
                self._anchor_continuity_loss(prepared, clean_probs, statistics)
                if continuity
                else None
            ),
            observed_axes=tuple(self.streams[prepared.domain]),
            anchor_neighborhood_loss=self._anchor_neighborhood_loss(
                prepared, clean_probs
            ),
            seam_fake=seam_fake,
            seam_domains=seam_domains,
            group_statistics_ramp=prepared.group_statistics_ramp,
        )

    def _transition_statistics(self, prepared):
        real = {
            axis: prepared.real.images[axis] for axis in self.streams[prepared.domain]
        }
        return transition_statistics(
            real,
            self.settings.connectivity_max_gap,
            prepared.real.heights if "height" in prepared.model_conditions else None,
            self.critic_augment.rotated_axes,
        )

    def _anchor_continuity_loss(self, prepared, clean_probs, statistics):
        selection = prepared.selection
        visible = prepared.presence.anchor
        loss, diagnostics = compute_anchor_continuity_loss(
            clean_probs,
            selection.measured.visible(visible),
            selection.condition.visible(visible),
            statistics,
            self.settings.connectivity_max_gap,
            prepared.model_conditions.get("height"),
        )
        self.diagnostics.update(diagnostics)
        return loss

    def _anchor_neighborhood_loss(self, prepared, clean_probs):
        selection = prepared.selection
        if selection is None:
            return None
        with torch.set_grad_enabled(
            torch.is_grad_enabled()
            and self.settings.loss.anchor_neighborhood_weight > 0
        ):
            loss, diagnostics = compute_anchor_neighborhood_loss(
                clean_probs,
                selection.condition.visible(prepared.presence.anchor),
                self.settings.loss.anchor_neighborhood_radius,
            )
        self.diagnostics.update(diagnostics)
        return loss

    def _real_transition_loss(self, prepared, clean_probs, statistics=None):
        real, fake, real_heights, fake_heights = {}, {}, {}, {}
        height = prepared.model_conditions.get("height")
        for axis in self.streams[prepared.domain]:
            if height is not None and axis not in prepared.real.heights:
                continue
            real[axis] = prepared.real.images[axis]
            # Sample before critic augmentation so directions retain physical meaning.
            sampled = sample_pairs(
                clean_probs,
                clean_probs,
                axis,
                self.settings.slice_pairs_per_axis,
                tuple(real[axis].shape[-2:]),
                height=height,
            )
            fake[axis] = sampled[0]
            if height is not None:
                real_heights[axis] = prepared.real.heights[axis]
                fake_heights[axis] = sampled[2]
        if not fake:
            return clean_probs.sum(dtype=torch.float32) * 0.0
        loss, diagnostics = compute_real_transition_loss(
            real,
            fake,
            self.settings.connectivity_max_gap,
            real_heights if height is not None else None,
            fake_heights if height is not None else None,
            statistics,
            self.critic_augment.rotated_axes,
        )
        self.diagnostics.update(diagnostics)
        return loss

    def make_connectivity_triplets(
        self,
        prediction: torch.Tensor,
        reference_prediction: torch.Tensor | None,
        anchor: AnchorCondition | None,
        transition: int,
        visible: torch.Tensor | None = None,
        source: str | None = None,
        height: torch.Tensor | None = None,
        has_profile: bool = False,
    ) -> tuple[TripletBatch, TripletBatch]:
        empty = TripletBatch(
            values=prediction.new_empty(
                (
                    0,
                    3,
                    self.settings.num_phases,
                    prediction.shape[-2],
                    prediction.shape[-1],
                )
            ),
            axes=torch.empty(0, device=self.device, dtype=torch.long),
            gaps=torch.empty(0, device=self.device, dtype=torch.long),
            center_slots=torch.empty(0, device=self.device, dtype=torch.long),
        )
        if (
            self.settings.loss.connectivity_weight <= 0.0
            or transition != 0
            or anchor is None
            or source != "multi"
            or reference_prediction is None
        ):
            return empty, empty
        anchor = anchor.visible(visible)
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
        height = None if real.height is None else augmented[2][:, :, :1]
        profile = augmented[2][:, :, 1:] if has_profile else None
        return (
            replace(real, values=augmented[0], height=height, profile=profile),
            replace(fake, values=augmented[1], height=height, profile=profile),
        )

    def make_seam_bands(
        self,
        prediction: torch.Tensor,
        anchor: AnchorCondition | None,
        visible: torch.Tensor,
    ) -> tuple[BandBatch, BandBatch]:
        enabled = (
            self.seam_critic is not None and self.settings.loss.anchor_seam_weight > 0.0
        )
        real, fake = sample_seam_bands(
            prediction,
            anchor.visible(visible) if enabled and anchor is not None else None,
            self.settings.loss.anchor_seam_rows,
            self.settings.patch_size // 2,
            self.settings.slice_pairs_per_axis,
            0 if self.settings.loss.preserve_height else None,
        )
        if enabled:
            self.diagnostics["anchor_seam/near_bands"] = len(fake)
            self.diagnostics["anchor_seam/far_bands"] = len(real)
        return real, fake

    @staticmethod
    def get_axis_domains(
        critic_domains: dict[int, int],
        axes: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(
            [critic_domains.get(axis, NULL_DOMAIN) for axis in AXES],
            device=axes.device,
            dtype=torch.long,
        )[axes]

    def regularization(self, name: str) -> tuple[bool, bool, bool]:
        """Lazy R1/R2 schedule for one critic's next update."""
        regularize = (self.updates[name] + 1) % self.settings.r1_interval == 0
        return (
            regularize,
            self.settings.r1_gamma > 0.0 and regularize,
            self.settings.r2_gamma > 0.0 and regularize,
        )

    def update_critics(
        self,
        transition: int,
        fake: SampledPairs,
        batches: RealBatch,
        domains: dict[int, int],
        profile_present: torch.Tensor | None = None,
    ) -> tuple[list[float], float]:
        """Return per-axis adversarial critic losses (R1/R2 excluded) and mean R1."""
        critic_losses = [0.0] * len(AXES)
        r1_sum = 0.0
        groups = active_groups(fake.pairs, self.critic_groups)
        for group, axes in groups.items():
            regularize, apply_r1, apply_r2 = self.regularization(group)
            weight = 1.0 / (len(axes) * len(groups))
            critic = self.critics[group]
            optimizer = self.critic_optims[group]
            optimizer.zero_grad(set_to_none=True)
            for axis in axes:
                real = batches.images[axis].mul(2.0).sub(1.0)
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
                real_domain = self.conditions.make_domain(
                    domains[axis], real_prev.shape[0]
                )
                fake_domain = self.conditions.make_domain(
                    domains[axis], fake_prev.shape[0]
                )

                autocast = self.autocast(self.settings.amp_enabled and not regularize)
                with autocast:
                    real_score = score_plane(
                        critic,
                        real_prev,
                        real_curr,
                        real_time,
                        real_domain,
                        **real_conditions,
                    )
                    fake_score = score_plane(
                        critic,
                        fake_prev,
                        fake_curr,
                        fake_time,
                        fake_domain,
                        **fake_conditions,
                    )
                    losses = critic_objective(
                        real_score,
                        fake_score,
                        (real_prev,),
                        (fake_prev,),
                        r1_weight=self.settings.r1_gamma if apply_r1 else 0.0,
                        r2_weight=self.settings.r2_gamma if apply_r2 else 0.0,
                        interval=self.settings.r1_interval,
                    )
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
                r1_sum += losses.r1.detach() * weight
                if apply_r2:
                    self.diagnostics[f"r2/{group}/{axis}"] = losses.r2.detach()
                loss = validate_loss(losses.total, f"critic {group}/{axis}")
                self.scaler.scale(loss * weight).backward()
                critic_losses[axis] = losses.adversarial.global_loss.detach() * weight
            if step_optimizer(optimizer, self.scaler, self.diagnostics, group):
                self.updates[group] += 1
        return critic_losses, r1_sum

    def update_connectivity_critic(
        self,
        real: TripletBatch,
        fake: TripletBatch,
        domains: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        if not len(fake):
            return 0.0, 0.0, 0.0, 0.0
        regularize, apply_r1, apply_r2 = self.regularization("connectivity")
        self.connectivity_optim.zero_grad(set_to_none=True)
        real_values = real.values.detach().float().requires_grad_(apply_r1)
        fake_values = fake.values.detach().float().requires_grad_(apply_r2)
        autocast = self.autocast(self.settings.amp_enabled and not regularize)
        with autocast:
            real_score = score_triplets(
                self.connectivity_critic, real, real_values, domains
            )
            fake_score = score_triplets(
                self.connectivity_critic, fake, fake_values, domains
            )
            losses = critic_objective(
                real_score,
                fake_score,
                (real_values,),
                (fake_values,),
                local_weight=self.settings.loss.connectivity_local_weight,
                r1_weight=self.settings.r1_gamma if apply_r1 else 0.0,
                r2_weight=self.settings.r2_gamma if apply_r2 else 0.0,
                interval=self.settings.r1_interval,
            )
        adversarial = losses.adversarial.combine(
            self.settings.loss.connectivity_local_weight
        ).detach()
        if apply_r2:
            self.diagnostics["r2/connectivity"] = losses.r2.detach()
        self.diagnostics["regularization/connectivity"] = int(regularize)
        loss = validate_loss(losses.total, "connectivity")
        self.scaler.scale(loss).backward()
        if step_optimizer(
            self.connectivity_optim, self.scaler, self.diagnostics, "connectivity"
        ):
            self.updates["connectivity"] += 1
        return (
            adversarial,
            losses.r1.detach(),
            losses.adversarial.global_loss.detach(),
            losses.adversarial.local_loss.detach(),
        )

    def update_seam_critic(
        self,
        transition: int,
        real: BandBatch,
        fake: BandBatch,
        domains: dict[int, int],
    ) -> None:
        if not len(fake):
            return
        regularize, apply_r1, apply_r2 = self.regularization("seam")
        self.seam_optim.zero_grad(set_to_none=True)
        real_values = real.values.detach().float().requires_grad_(apply_r1)
        fake_values = fake.values.detach().float().requires_grad_(apply_r2)
        with self.autocast(self.settings.amp_enabled and not regularize):
            real_score = self.seam_critic(
                real_values,
                real.axes,
                self.make_time(transition, len(real)),
                self.get_axis_domains(domains, real.axes),
            )
            fake_score = self.seam_critic(
                fake_values,
                fake.axes,
                self.make_time(transition, len(fake)),
                self.get_axis_domains(domains, fake.axes),
            )
            losses = critic_objective(
                real_score,
                fake_score,
                (real_values,),
                (fake_values,),
                r1_weight=self.settings.r1_gamma if apply_r1 else 0.0,
                r2_weight=self.settings.r2_gamma if apply_r2 else 0.0,
                interval=self.settings.r1_interval,
            )
        self.diagnostics["loss/critic_anchor_seam"] = (
            losses.adversarial.global_loss.detach()
        )
        self.diagnostics["anchor_seam/far_score"] = (
            real_score.logits_global.detach().mean()
        )
        self.diagnostics["anchor_seam/near_score"] = (
            fake_score.logits_global.detach().mean()
        )
        self.diagnostics["anchor_seam/r1"] = losses.r1.detach()
        if apply_r2:
            self.diagnostics["r2/anchor_seam"] = losses.r2.detach()
        loss = validate_loss(losses.total, "anchor seam")
        self.scaler.scale(loss).backward()
        if step_optimizer(
            self.seam_optim, self.scaler, self.diagnostics, "anchor_seam"
        ):
            self.updates["seam"] += 1

    def update_denoiser(self, batch: DenoiserBatch) -> DenoiserUpdate:
        self.denoiser_optim.zero_grad(set_to_none=True)
        critics = self.critic_modules()
        for critic in critics:
            critic.requires_grad_(False)
        try:
            with self.autocast():
                losses, diagnostics = denoiser_objective(
                    batch,
                    self.critics,
                    self.connectivity_critic,
                    self.seam_critic,
                    self.critic_groups,
                    self.anchor_loss,
                    self.settings.loss,
                    diagnose=(self.updates["generator"] + 1) % self.settings.r1_interval
                    == 0,
                )
            self.diagnostics.update(diagnostics)
            total = validate_loss(losses.total, "generator")
            self.scaler.scale(total).backward()
            self.generator_updated = step_optimizer(
                self.denoiser_optim, self.scaler, self.diagnostics, "generator"
            )
            if self.generator_updated:
                self.updates["generator"] += 1
        finally:
            for critic in critics:
                critic.requires_grad_(True)
        return losses.detach()

    def critic_modules(self) -> list[nn.Module]:
        extra = (self.connectivity_critic, self.seam_critic)
        return [*self.critics.values(), *(m for m in extra if m is not None)]

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
            self.settings.latent_channels,
            device=self.device,
            dtype=dtype,
        )

    def autocast(
        self,
        enabled: bool | None = None,
    ) -> AbstractContextManager:
        if enabled is None:
            enabled = self.settings.amp_enabled
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=enabled,
        )
