from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from src.data.augment import CriticAugment
from src.data.loader import BatchStream
from src.data.slice import sample_pairs
from src.evaluate.structure import structure_metrics
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.plane import AXES, PLANE_AXES
from src.train.batch import (
    DenoiserBatch,
    DenoiserUpdate,
    RealBatch,
    SampledPairs,
    StepPreparation,
)
from src.train.burn_in import BurnIn, BurnInValues
from src.train.conditioning import ConditionSampler
from src.train.ema import update_ema
from src.train.loss.denoiser import DenoiserLossSettings, denoiser_objective
from src.train.loss.gan import (
    active_groups,
    critic_objective,
    get_relativistic_loss,
    score_plane,
)
from src.train.loss.transition import compute_real_transition_loss
from src.train.metrics import Metrics, collect_metrics
from src.train.selection import CheckpointSelection
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
    streams: dict[int, dict[int, BatchStream]]
    diffusion: Diffusion
    denoiser_optim: torch.optim.Optimizer
    critic_optims: dict[str, torch.optim.Optimizer]
    scaler: torch.amp.GradScaler
    device: torch.device
    critic_augment: CriticAugment | None = None
    coarse_bank: dict | None = None
    bank_origins: dict | None = None
    bank_extents: dict | None = None
    critic_groups_by_domain: dict | None = None
    selection: CheckpointSelection | None = None
    data_fingerprint: dict[str, str] = field(default_factory=dict)
    cfg: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerSettings:
    """Runtime settings; src/build/trainer.py fills them from the resolved config."""

    volume_batch_size: int
    num_phases: int
    patch_size: int
    latent_channels: int
    slice_pairs_per_plane: int
    burn_in: BurnIn
    r1_weight: float
    r2_weight: float
    r1_every_steps: int
    loss: DenoiserLossSettings
    group_statistics_start_step: int
    group_statistics_ramp_steps: int
    real_transition_max_gap: int
    real_transition_start_step: int
    real_transition_ramp_steps: int
    structure_every_steps: int
    cfg_dropout_probability: float
    domain_keep_probability: float
    coarse_corruption_probability: float
    coarse_corruption_strength: float
    height_data: dict | None
    profile_settings: dict
    volume_permute_z: bool
    volume_flip_z: bool
    # fp16 for the denoiser, under the GradScaler; on CUDA and MPS.
    denoiser_amp: bool
    # fp16 for the critics, on CUDA only: on MPS (torch 2.14) it faults the GPU.
    critic_amp: bool


class Trainer:
    def __init__(
        self,
        components: TrainerComponents,
        settings: TrainerSettings,
    ) -> None:
        self.settings = settings
        self.scheduled = settings.burn_in.values(0)
        self.denoiser = components.denoiser
        self.ema_denoiser = components.ema_denoiser
        self.critics = components.critics
        self.streams = components.streams
        self.data_fingerprint = components.data_fingerprint
        self.selection = components.selection
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
        self.scaler = components.scaler
        self.device = components.device
        self.updates = {name: 0 for name in (*self.critics, "generator")}
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
        self.active_axes = tuple(
            sorted(axis for axes in groups.values() for axis in axes)
        )

    def step(
        self,
        step: int,
        transition: int | None = None,
    ) -> Metrics:
        self.diagnostics = {}
        self.scheduled = self.apply_burn_in(step)
        prepared = self.prepare_step(step, transition)
        transition = prepared.transition
        real = prepared.real
        previous, current, prediction = self.generate_pair(
            transition, prepared.model_conditions
        )
        clean_probs = (prediction + 1.0) * 0.5
        fake = self._sample_fake_pairs(
            previous, current, real, prepared.model_conditions.get("height")
        )
        real_pairs = self._sample_real_pairs(transition, real)
        critic_vals, r1 = self.update_critics(
            transition, fake, real_pairs, prepared.critic_domains
        )
        denoiser_update = self.update_denoiser(
            self._make_denoiser_batch(prepared, fake, real_pairs, clean_probs)
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
            update_ema(self.ema_denoiser, self.denoiser, self.scheduled.ema_decay)
        return collect_metrics(
            diagnostics=self.diagnostics,
            prepared=prepared,
            denoiser_update=denoiser_update,
            critic_vals=critic_vals,
            r1=r1,
        )

    def prepare_step(
        self,
        step: int,
        transition: int | None,
    ) -> StepPreparation:
        self.denoiser.train()
        self.critics.train()

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.settings.patch_size
        shape = (
            self.settings.volume_batch_size,
            self.settings.num_phases,
            size,
            size,
            size,
        )
        current = torch.randn(shape, device=self.device, dtype=torch.float32)

        with torch.no_grad(), self.autocast(self.settings.denoiser_amp):
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
        with self.autocast(self.settings.denoiser_amp):
            prediction = self.denoiser(current, time, latent, **model_conditions)
            previous = self.diffusion.sample_posterior(
                current,
                prediction,
                transition,
            )
        return previous, current, prediction

    def _sample_fake_pairs(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        real: RealBatch,
        volume_height: torch.Tensor | None,
    ) -> SampledPairs:
        result = SampledPairs(pairs={})
        for axis in self.active_axes:
            height = volume_height if axis in real.heights else None
            pairs = sample_pairs(
                previous,
                current,
                axis,
                self.settings.slice_pairs_per_plane,
                tuple(real.images[axis].shape[-2:]),
                height=height,
            )
            if not self.critic_augment.real_only:
                pairs = self.critic_augment.apply_together(pairs, plane=axis)
            result.add(axis, pairs)
        return result

    def _sample_real_pairs(
        self,
        transition: int,
        real: RealBatch,
    ) -> SampledPairs:
        result = SampledPairs(pairs={})
        for axis in self.active_axes:
            images = real.images[axis].mul(2.0).sub(1.0)
            previous, current = self.diffusion.sample_pair(images, transition)
            height = real.heights.get(axis)
            inputs = (
                (previous, current) if height is None else (previous, current, height)
            )
            result.add(axis, self.critic_augment.apply_together(inputs, plane=axis))
        return result

    def _make_denoiser_batch(
        self,
        prepared: StepPreparation,
        fake: SampledPairs,
        real: SampledPairs,
        clean_probs: torch.Tensor,
    ) -> DenoiserBatch:
        real_transition = (
            self.settings.loss.real_transition_weight > 0 and prepared.transition == 0
        )
        return DenoiserBatch(
            transition=prepared.transition,
            critic_domains=prepared.critic_domains,
            fake=fake,
            real=real,
            clean_probs=clean_probs,
            real_transition_ramp=prepared.real_transition_ramp,
            target_vf=prepared.target_vf,
            vf_present=prepared.vf_present,
            profile=prepared.model_conditions.get("profile"),
            coarse_target=prepared.coarse_target,
            coarse_mask=prepared.coarse_mask,
            real_transition_loss=(
                self._real_transition_loss(prepared, clean_probs)
                if real_transition
                else None
            ),
            observed_axes=tuple(self.streams[prepared.domain]),
            group_statistics_ramp=prepared.group_statistics_ramp,
        )

    def _real_transition_loss(self, prepared, clean_probs):
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
                self.settings.slice_pairs_per_plane,
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
            self.settings.real_transition_max_gap,
            real_heights if height is not None else None,
            fake_heights if height is not None else None,
            self.critic_augment.rotated_axes,
        )
        self.diagnostics.update(diagnostics)
        return loss

    def apply_burn_in(self, step: int) -> BurnInValues:
        """Set this step's learning rates and β2; return its scheduled values."""
        values = self.settings.burn_in.values(step)
        for optimizer, lr in (
            (self.denoiser_optim, values.generator_lr),
            *((optim, values.critic_lr) for optim in self.critic_optims.values()),
        ):
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["betas"] = values.betas
        self.diagnostics.update(
            {
                "burn_in/generator_lr": values.generator_lr,
                "burn_in/critic_lr": values.critic_lr,
                "burn_in/beta2": values.betas[1],
                "burn_in/regularization_scale": values.regularization_scale,
                "burn_in/ema_decay": values.ema_decay,
            }
        )
        return values

    def regularization(self, name: str) -> tuple[bool, float, float]:
        """Lazy R1/R2 schedule and weights for one critic's next update."""
        regularize = (self.updates[name] + 1) % self.settings.r1_every_steps == 0
        scale = self.scheduled.regularization_scale if regularize else 0.0
        return (
            regularize,
            self.settings.r1_weight * scale,
            self.settings.r2_weight * scale,
        )

    def update_critics(
        self,
        transition: int,
        fake: SampledPairs,
        real: SampledPairs,
        domains: dict[int, int],
    ) -> tuple[list[float], float]:
        """Return per-axis adversarial critic losses (R1/R2 excluded) and mean R1."""
        critic_losses = [0.0] * len(AXES)
        r1_sum = 0.0
        groups = active_groups(fake.pairs, self.critic_groups)
        for group, axes in groups.items():
            regularize, r1_weight, r2_weight = self.regularization(group)
            weight = 1.0 / (len(axes) * len(groups))
            critic = self.critics[group]
            optimizer = self.critic_optims[group]
            optimizer.zero_grad(set_to_none=True)
            for axis in axes:
                # Copies keep the shared real pairs free of gradients for the generator.
                real_prev, real_curr = (
                    value.detach().requires_grad_(regularize)
                    for value in real.pairs[axis]
                )
                real_time = self.make_time(transition, real_prev.shape[0])
                fake_prev, fake_curr = fake.pairs[axis]
                fake_prev = fake_prev.detach().float().requires_grad_(r2_weight > 0)
                fake_curr = fake_curr.detach().float()
                fake_time = self.make_time(transition, fake_prev.shape[0])
                real_domain = self.conditions.make_domain(
                    domains[axis], real_prev.shape[0]
                )
                fake_domain = self.conditions.make_domain(
                    domains[axis], fake_prev.shape[0]
                )

                autocast = self.autocast(self.settings.critic_amp and not regularize)
                with autocast:
                    real_score = score_plane(
                        critic,
                        real_prev,
                        real_curr,
                        real_time,
                        real_domain,
                        **real.conditions(axis),
                    )
                    fake_score = score_plane(
                        critic,
                        fake_prev,
                        fake_curr,
                        fake_time,
                        fake_domain,
                        **fake.conditions(axis),
                    )
                    # Pairs need not share a height: fake volumes take their heights
                    # from real crops, so each pair compares (section, height) samples.
                    losses = critic_objective(
                        get_relativistic_loss(real_score, fake_score),
                        real_score,
                        fake_score,
                        (real_prev,),
                        (fake_prev,),
                        r1_weight=r1_weight,
                        r2_weight=r2_weight,
                        interval=self.settings.r1_every_steps,
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
                if r2_weight > 0:
                    self.diagnostics[f"r2/{group}/{axis}"] = losses.r2.detach()
                loss = validate_loss(losses.total, f"critic {group}/{axis}")
                self.scaler.scale(loss * weight).backward()
                critic_losses[axis] = losses.adversarial.detach() * weight
            if step_optimizer(optimizer, self.scaler, self.diagnostics, group):
                self.updates[group] += 1
        return critic_losses, r1_sum

    def update_denoiser(self, batch: DenoiserBatch) -> DenoiserUpdate:
        self.denoiser_optim.zero_grad(set_to_none=True)
        self.critics.requires_grad_(False)
        try:
            # The objective scores fake pairs with the critics.
            with self.autocast(self.settings.critic_amp):
                losses, diagnostics = denoiser_objective(
                    batch,
                    self.critics,
                    self.critic_groups,
                    self.settings.loss,
                    diagnose=(self.updates["generator"] + 1)
                    % self.settings.r1_every_steps
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
            self.critics.requires_grad_(True)
        return losses.detach()

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

    def autocast(self, enabled: bool) -> AbstractContextManager:
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=enabled,
        )
