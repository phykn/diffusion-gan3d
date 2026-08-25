import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import AXES
from .anchor import AnchorCondition, PlaneAnchor, encode_anchors
from .dataset import BatchStream
from .dataset.augment import CriticAugment
from .loss import vf
from .loss.anchor import SoftAnchorLoss
from .loss.connect import AnchorTripletSampler, TripletBatch, compute_transition_loss
from .loss.gan import (
    get_critic_loss,
    get_critic_r1,
    get_generator_loss,
)
from .model.common import NULL_DOMAIN
from .model.denoiser import Denoiser3D
from .model.diffusion import Diffusion
from .model.ema import update_ema
from .utils import save_model


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
    vf: float


@dataclass(frozen=True)
class DenoiserBatch:
    transition: int
    connectivity_domains: torch.Tensor
    critic_domains: dict[int, int]
    fake: dict[int, tuple[torch.Tensor, torch.Tensor]]
    connectivity_real: TripletBatch
    connectivity_fake: TripletBatch
    logits: torch.Tensor
    clean_probs: torch.Tensor
    anchor: AnchorCondition | None
    anchor_observed_mask: torch.Tensor | None
    anchor_observed_axis_masks: torch.Tensor | None
    anchor_present: torch.Tensor
    anchor_ramp: float
    target_vf: torch.Tensor
    vf_present: torch.Tensor


@dataclass(frozen=True)
class StepPreparation:
    transition: int
    domain: int
    critic_domains: dict[int, int]
    real: dict[int, torch.Tensor]
    selection: "AnchorSelection | None"
    target_vf: torch.Tensor
    presence: "ConditionPresence"
    model_conditions: dict[str, torch.Tensor]
    anchor_ramp: float


@dataclass(frozen=True)
class AnchorSelection:
    condition: AnchorCondition | None
    observed_mask: torch.Tensor | None
    observed_axis_masks: torch.Tensor | None
    source: Literal["real", "shared", "multi"]


@dataclass(frozen=True)
class ConditionPresence:
    anchor: torch.Tensor
    vf: torch.Tensor


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
        self.anchor_shared_axis_probability = float(
            settings.anchor_shared_axis_probability
        )
        pool_exponent = math.ceil(
            math.log2(self.denoiser.downsample_factor) / 2.0
        )
        self.anchor_loss = SoftAnchorLoss(
            pool_size=2**pool_exponent,
            pixel_weight=settings.anchor_pixel_loss_weight,
        )
        self.connectivity_weight = settings.connectivity_weight
        self.normal_transition_weight = settings.normal_transition_weight
        self.anchor_triplets = AnchorTripletSampler(
            max_gap=settings.connectivity_max_gap,
        )
        self.use_multi_anchor_next = False
        self.vf_loss_weight = settings.vf_loss_weight
        self.cfg_drop_each_probability = float(settings.cfg_drop_each_probability)
        self.domain_dropout = float(settings.domain_dropout)
        self.latent_channels = settings.latent_channels
        self.amp_enabled = settings.amp_enabled
        self.critic_augment = (
            CriticAugment(False)
            if components.critic_augment is None
            else components.critic_augment
        )

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
        presence = prepared.presence
        selection = prepared.selection
        needs_reference = selection is not None and (
            selection.source == "multi"
            or (
                selection.source in ("real", "shared")
                and (
                    self.connectivity_weight > 0.0
                    or self.normal_transition_weight > 0.0
                )
            )
        )
        reference = None
        if needs_reference:
            rng_state = self.capture_rng_state()
            with torch.no_grad():
                reference = self.generate_pair(
                    transition,
                    self.remove_anchor_conditions(prepared.model_conditions),
                    volume_size,
                )
            if selection.source == "multi":
                selection = self.sample_multi_anchor(reference[3])
                prepared = replace(
                    prepared,
                    selection=selection,
                    model_conditions=self.make_model_conditions(
                        selection.condition,
                        prepared.target_vf,
                        prepared.presence,
                        prepared.model_conditions["domain"],
                    ),
                )
            self.restore_rng_state(rng_state)
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
        fake = self._sample_fake_pairs(
            previous,
            current,
            real,
            anchor,
            presence.anchor,
        )

        reference_pairs = None
        if selection is not None and selection.source == "multi":
            assert reference is not None
            reference_pairs = self._sample_fake_pairs(
                reference[0],
                reference[1],
                real,
                None,
                torch.zeros_like(presence.anchor),
            )
        connectivity_real, connectivity_fake = self.make_connectivity_triplets(
            prediction,
            None if reference is None else reference[3],
            anchor,
            transition,
            presence.anchor,
            None if selection is None else selection.source,
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
            real_pairs=reference_pairs,
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
        real: dict[int, torch.Tensor],
        anchor: AnchorCondition | None,
        anchor_present: torch.Tensor,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        visible_axis_masks = (
            None
            if anchor is None
            else anchor.axis_masks & anchor_present.reshape(-1, 1, 1, 1, 1)
        )
        return {
            axis: self.critic_augment.apply_together(
                self.sample_pairs(
                    previous,
                    current,
                    axis,
                    axis_masks=visible_axis_masks,
                    crop_shape=tuple(real[axis].shape[-2:]),
                ),
            )
            for axis in self.active_axes
        }

    @staticmethod
    def _make_denoiser_batch(
        prepared: StepPreparation,
        connectivity_domains: torch.Tensor,
        fake: dict[int, tuple[torch.Tensor, torch.Tensor]],
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
            fake=fake,
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
            target_vf=prepared.target_vf,
            vf_present=prepared.presence.vf,
        )

    def prepare_step(
        self,
        step: int,
        transition: int | None,
    ) -> StepPreparation:
        self.denoiser.train()
        self.critics.train()
        self.connectivity_critic.train()

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
        target_vf = vf.compute_vf(own_batches, self.num_phases)
        target_vf = target_vf.unsqueeze(0).expand(self.volume_batch_size, -1)
        ramp = self.get_anchor_ramp(step)
        selection = (
            None
            if ramp == 0.0
            else self.sample_anchor(
                batches,
                self.patch_size,
                owned_axes=tuple(own_batches),
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
        update_ema(self.ema_denoiser, self.denoiser, self.ema_decay)
        selection = prepared.selection
        anchor = None if selection is None else selection.condition
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
            generator_global=denoiser_update.global_loss,
            generator_local=denoiser_update.local_loss,
            critic_global=critic_global,
            critic_local=critic_local,
            vf_loss=denoiser_update.vf,
            vf_active=bool(presence.vf.any()),
            anchor_input_active_fraction=float(
                presence.anchor.to(torch.float32).mean()
            ),
            vf_active_fraction=float(presence.vf.to(torch.float32).mean()),
            normal_transition_loss=denoiser_update.normal_transition,
            anchor_coarse_loss=denoiser_update.anchor_coarse,
            anchor_pixel_loss=denoiser_update.anchor_pixel,
            anchor_shared=(
                prepared.selection is not None
                and prepared.selection.source == "shared"
                and bool(presence.anchor.any())
            ),
        )

    def make_connectivity_triplets(
        self,
        prediction: torch.Tensor,
        reference_prediction: torch.Tensor | None,
        anchor: AnchorCondition | None,
        transition: int,
        visible: torch.Tensor | None = None,
        source: str | None = None,
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
            or source not in ("real", "shared")
            or reference_prediction is None
        ):
            return empty, empty
        anchor = self.visible_anchor(anchor, visible)
        if not bool(anchor.mask.any()):
            return empty, empty

        real, fake = self.anchor_triplets.sample(
            prediction,
            reference_prediction,
            anchor,
        )
        real_values, fake_values = self.critic_augment.apply_together(
            (real.values, fake.values),
        )
        return (
            TripletBatch(
                values=real_values,
                axes=real.axes,
                gaps=real.gaps,
                center_slots=real.center_slots,
            ),
            TripletBatch(
                values=fake_values,
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
            [critic_domains.get(axis, NULL_DOMAIN) for axis in triplets.axes.tolist()],
            device=triplets.values.device,
            dtype=torch.long,
        )

    @staticmethod
    def visible_anchor(
        anchor: AnchorCondition,
        visible: torch.Tensor | None,
    ) -> AnchorCondition:
        if visible is None:
            return anchor
        mask = visible.reshape(-1, 1, 1, 1, 1)
        return replace(
            anchor,
            image=anchor.image * mask,
            mask=anchor.mask & mask,
            axis_masks=anchor.axis_masks & mask,
        )

    def sample_transition(self, anchored: bool) -> int:
        if not anchored:
            return int(torch.randint(self.diffusion.timesteps, ()).item())
        if self.diffusion.timesteps == 1 or bool(torch.rand(()) < 0.25):
            return 0
        return int(torch.randint(1, self.diffusion.timesteps, ()).item())

    @staticmethod
    def remove_anchor_conditions(
        conditions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in conditions.items()
            if name not in {"anchor_image", "anchor_mask"}
        }

    def capture_rng_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device_state = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        return torch.random.get_rng_state(), device_state

    def restore_rng_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        cpu_state, device_state = state
        torch.random.set_rng_state(cpu_state)
        if device_state is not None:
            torch.cuda.set_rng_state(device_state, self.device)

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
        crop_h, crop_w = (size, size) if isinstance(size, int) else size
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
            for index, (row, col) in enumerate(centers):
                top[index] = min(max(row - crop_h // 2, 0), height - crop_h)
                left[index] = min(max(col - crop_w // 2, 0), width - crop_w)
        return torch.stack(
            [
                image[..., row : row + crop_h, col : col + crop_w]
                for image, row, col in zip(images, top, left, strict=True)
            ]
        )

    def sample_condition_presence(self, has_anchor: bool) -> ConditionPresence:
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
        if self.anchor_ramp_steps == 0:
            return 1.0
        return min(
            (step - self.anchor_start_step + 1) / self.anchor_ramp_steps,
            1.0,
        )

    def sample_anchor(
        self,
        batches: dict[int, torch.Tensor],
        volume_size: int,
        owned_axes: tuple[int, ...] | None = None,
    ) -> AnchorSelection | None:
        probability = self.anchor_training_probability
        if probability <= 0.0:
            return None
        if probability < 1.0 and not bool(torch.rand(()) < probability):
            return None
        if self.use_multi_anchor_next:
            self.use_multi_anchor_next = False
            return AnchorSelection(
                condition=None,
                observed_mask=None,
                observed_axis_masks=None,
                source="multi",
            )
        self.use_multi_anchor_next = True
        return self.sample_real_anchor(
            batches,
            volume_size,
            owned_axes=owned_axes,
        )

    def sample_real_anchor(
        self,
        batches: dict[int, torch.Tensor],
        volume_size: int,
        owned_axes: tuple[int, ...] | None = None,
    ) -> AnchorSelection:
        if owned_axes is None:
            owned_axes = tuple(batches)
        shared_axes = tuple(axis for axis in batches if axis not in owned_axes)
        use_shared = (
            bool(shared_axes)
            and self.anchor_shared_axis_probability > 0.0
            and bool(torch.rand(()) < self.anchor_shared_axis_probability)
        )
        axes = shared_axes if use_shared else owned_axes
        axis = axes[int(torch.randint(len(axes), ()).item())]
        images = batches[axis]
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
        condition = encode_anchors(
            (plane,),
            batch_size=self.volume_batch_size,
            num_phases=self.num_phases,
            volume_size=volume_size,
            device=self.device,
            dtype=torch.float32,
            reconcile=False,
        )
        return AnchorSelection(
            condition=condition,
            observed_mask=condition.mask,
            observed_axis_masks=condition.axis_masks,
            source="shared" if use_shared else "real",
        )

    def sample_multi_anchor(self, prediction: torch.Tensor) -> AnchorSelection:
        labels = prediction.detach().argmax(dim=1)
        count = int(torch.randint(2, len(AXES) + 1, ()).item())
        axes = torch.randperm(len(AXES), device=prediction.device)[:count].tolist()
        planes = []
        for axis in axes:
            index = int(
                torch.randint(
                    prediction.shape[axis + 2],
                    (),
                    device=prediction.device,
                ).item()
            )
            planes.append(
                PlaneAnchor(
                    image=labels.select(axis + 1, index),
                    axis=axis,
                    index=index,
                )
            )
        condition = encode_anchors(
            tuple(planes),
            batch_size=prediction.shape[0],
            num_phases=self.num_phases,
            volume_size=prediction.shape[2],
            device=prediction.device,
            dtype=torch.float32,
            reconcile=False,
        )
        return AnchorSelection(
            condition=condition,
            observed_mask=torch.zeros_like(condition.mask),
            observed_axis_masks=torch.zeros_like(condition.axis_masks),
            source="multi",
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

    def sample_pairs(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        axis: int,
        axis_masks: torch.Tensor | None = None,
        crop_shape: int | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        real_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
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

            if real_pairs is None:
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
                real_prev, real_curr = self.critic_augment.apply_together(
                    (real_prev, real_curr),
                )
            else:
                real_prev, real_curr = real_pairs[axis]
                real_time = self.make_time(transition, real_prev.shape[0])
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
        domains: torch.Tensor,
    ) -> tuple[float, float]:
        if not len(fake):
            return 0.0, 0.0

        apply_r1 = self.r1_gamma > 0.0 and (step + 1) % self.r1_interval == 0
        self.connectivity_optim.zero_grad(set_to_none=True)
        real = real.detach().float().requires_grad_(apply_r1)
        fake_values = fake.values.detach().float()
        autocast = self.autocast(self.amp_enabled and not apply_r1)
        with autocast:
            real_score = self.connectivity_critic(
                real,
                fake.axes,
                fake.gaps,
                domains,
            )
            fake_score = self.connectivity_critic(
                fake_values,
                fake.axes,
                fake.gaps,
                domains,
            )
            losses = get_critic_loss(
                real_score,
                fake_score,
            )
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
                        batch.connectivity_fake.gaps,
                        batch.connectivity_domains,
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
                vf_loss = vf.compute_vf_loss(
                    batch.clean_probs,
                    batch.target_vf,
                    batch.vf_present,
                )
                total = (
                    adversarial_loss
                    + batch.anchor_ramp
                    * (
                        self.connectivity_weight * connectivity_loss
                        + self.normal_transition_weight * normal_loss
                        + anchor_loss
                    )
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
            vf=float(vf_loss.detach()),
        )

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


def run_train(
    trainer: Trainer,
    steps: int,
    save_every: int,
    run_dir: str | Path,
    checkpoint_every: int | None = None,
) -> Path:
    root = Path(run_dir)
    done = 0
    weights = root / "generator.pt"
    writer = SummaryWriter(root / "tensorboard")
    bar = tqdm(
        range(steps),
        desc="Diffusion GAN3D",
        dynamic_ncols=True,
    )
    model_files = {
        "generator.pt": trainer.ema_denoiser,
        **{
            f"critic_{axis}.pt": critic
            for axis, critic in trainer.critics.items()
        },
        "critic_c.pt": trainer.connectivity_critic,
    }
    try:
        for step in bar:
            metrics = trainer.step(step)
            done = step + 1
            write_metrics(writer, done, metrics)
            if done % save_every == 0:
                for name, model in model_files.items():
                    save_model(root / name, model)
            if checkpoint_every is not None and done % checkpoint_every == 0:
                checkpoint_root = root / "checkpoints" / f"step_{done:08d}"
                for name, model in model_files.items():
                    save_model(checkpoint_root / name, model)
        if done % save_every:
            for name, model in model_files.items():
                save_model(root / name, model)
    except KeyboardInterrupt:
        for name, model in model_files.items():
            save_model(root / name, model)
        raise
    finally:
        bar.close()
        writer.close()
    return weights


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
    }

    if metrics.anchor_planes:
        scalars["conditioning/anchor_planes"] = metrics.anchor_planes
        scalars["conditioning/anchor_accuracy"] = metrics.anchor_accuracy

    for tag, value in scalars.items():
        writer.add_scalar(tag, value, step)
