from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..anchor import AnchorCondition, PlaneAnchor, build_anchors
from ..data import (
    AXES,
    BatchStream,
    encode_labels,
    sample_pairs,
)
from ..diffusion import Diffusion
from ..model import Denoiser3D
from .config import TrainConfig
from .ema import update_ema
from .loss import critic_logistic_loss, generator_logistic_loss, r1_penalty


@dataclass(frozen=True)
class Metrics:
    generator: float
    generator_total: float
    critic: float
    r1: float
    transition: int
    critic_axes: tuple[float, float, float]
    anchor_used: bool
    anchor_loss: float
    anchor_accuracy: float


class Trainer:
    def __init__(
        self,
        *,
        denoiser: Denoiser3D,
        ema_denoiser: Denoiser3D,
        critics: nn.ModuleDict,
        streams: dict[int, BatchStream],
        diffusion: Diffusion,
        denoiser_optimizer: torch.optim.Optimizer,
        critic_optimizers: dict[str, torch.optim.Optimizer],
        scaler: torch.amp.GradScaler,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        if set(streams) != set(AXES):
            raise ValueError("streams must contain axes 0, 1, and 2.")
        if set(critic_optimizers) != {str(axis) for axis in AXES}:
            raise ValueError("critic optimizers must contain axes 0, 1, and 2.")
        self.denoiser = denoiser
        self.ema_denoiser = ema_denoiser
        self.critics = critics
        self.streams = streams
        self.diffusion = diffusion
        self.denoiser_optimizer = denoiser_optimizer
        self.critic_optimizers = critic_optimizers
        self.scaler = scaler
        self.cfg = cfg
        self.amp_enabled = cfg.train.mixed_precision and device.type == "cuda"
        self.device = device

    def step(
        self,
        step: int,
        *,
        transition: int | None = None,
    ) -> Metrics:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer.")
        self.denoiser.train()
        self.critics.train()
        if transition is None:
            transition = int(
                torch.randint(
                    self.diffusion.timesteps,
                    (1,),
                    device=self.device,
                ).item()
            )
        elif (
            not isinstance(transition, int)
            or isinstance(transition, bool)
            or not 0 <= transition < self.diffusion.timesteps
        ):
            raise ValueError("transition is outside the diffusion schedule.")
        anchor = self._sample_anchor()
        previous_volume, current_volume, clean_logits = self._generate_pair(
            transition,
            anchor,
        )
        fake_pairs = {
            axis: sample_pairs(
                previous_volume,
                current_volume,
                axis=axis,
                count=self.cfg.train.slices_per_axis,
            )
            for axis in AXES
        }

        critic_values, r1_value = self._update_critics(
            transition,
            fake_pairs,
            step,
        )
        (
            generator_value,
            generator_total,
            anchor_loss,
            anchor_accuracy,
        ) = self._update_denoiser(
            transition,
            fake_pairs,
            clean_logits,
            anchor,
        )
        update_ema(self.ema_denoiser, self.denoiser, self.cfg.train.ema_decay)
        return Metrics(
            generator=generator_value,
            generator_total=generator_total,
            critic=sum(critic_values),
            r1=r1_value,
            transition=transition,
            critic_axes=tuple(critic_values),
            anchor_used=anchor is not None,
            anchor_loss=anchor_loss,
            anchor_accuracy=anchor_accuracy,
        )

    def _generate_pair(
        self,
        transition: int,
        anchor: AnchorCondition | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (
            self.cfg.train.volume_batch_size,
            self.cfg.data.num_phases,
            self.cfg.data.patch_size,
            self.cfg.data.patch_size,
            self.cfg.data.patch_size,
        )
        current = torch.randn(shape, device=self.device, dtype=torch.float32)
        with torch.no_grad(), self._autocast():
            for index in reversed(range(transition + 1, self.diffusion.timesteps)):
                times = self._make_times(index, current.shape[0])
                latent = self._sample_latent(current.shape[0], dtype=current.dtype)
                clean = self._denoise(current, times, latent, anchor)
                current = self.diffusion.sample_posterior(
                    current,
                    clean,
                    times,
                )

        current = current.detach()
        times = self._make_times(transition, current.shape[0])
        latent = self._sample_latent(current.shape[0], dtype=current.dtype)
        with self._autocast():
            clean_logits = self._predict_logits(
                current,
                times,
                latent,
                anchor,
            )
            clean = self.denoiser.decode(clean_logits)
            previous = self.diffusion.sample_posterior(
                current,
                clean,
                times,
            )
        return previous, current, clean_logits

    def _update_critics(
        self,
        transition: int,
        fake_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> tuple[list[float], float]:
        apply_r1 = (
            self.cfg.optim.r1_gamma > 0.0
            and (step + 1) % self.cfg.optim.r1_interval == 0
        )
        critic_values = []
        total_r1 = 0.0
        for axis in AXES:
            critic = self.critics[str(axis)]
            optimizer = self.critic_optimizers[str(axis)]
            optimizer.zero_grad(set_to_none=True)

            labels = (
                self.streams[axis]
                .next()
                .to(
                    self.device,
                    non_blocking=True,
                )
            )
            real_clean = encode_labels(labels, self.cfg.data.num_phases)
            real_times = self._make_times(transition, real_clean.shape[0])
            real_previous, real_current = self.diffusion.sample_pair(
                real_clean,
                real_times,
            )
            real_previous.requires_grad_(apply_r1)
            fake_previous, fake_current = fake_pairs[axis]
            fake_previous = fake_previous.detach().float()
            fake_current = fake_current.detach().float()
            fake_times = self._make_times(transition, fake_previous.shape[0])

            autocast = self._autocast(enabled=self.amp_enabled and not apply_r1)
            with autocast:
                real_logits = critic(real_previous, real_current, real_times)
                fake_logits = critic(fake_previous, fake_current, fake_times)
                loss = critic_logistic_loss(real_logits, fake_logits)
            if apply_r1:
                penalty = r1_penalty(
                    real_logits,
                    (real_previous,),
                )
                total_r1 += float(penalty.detach())
                loss = (
                    loss
                    + 0.5
                    * self.cfg.optim.r1_gamma
                    * self.cfg.optim.r1_interval
                    * penalty
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            critic_values.append(float(loss.detach()))
        self.scaler.update()
        return critic_values, total_r1

    def _update_denoiser(
        self,
        transition: int,
        fake_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]],
        clean_logits: torch.Tensor,
        anchor: AnchorCondition | None,
    ) -> tuple[float, float, float, float]:
        self.denoiser_optimizer.zero_grad(set_to_none=True)
        for critic in self.critics.values():
            critic.requires_grad_(False)
        try:
            losses = []
            with self._autocast():
                for axis in AXES:
                    fake_previous, fake_current = fake_pairs[axis]
                    times = self._make_times(transition, fake_previous.shape[0])
                    logits = self.critics[str(axis)](
                        fake_previous,
                        fake_current,
                        times,
                    )
                    losses.append(generator_logistic_loss(logits))
                adversarial = torch.stack(losses).sum()
                anchor_loss = adversarial.new_zeros(())
                anchor_accuracy = adversarial.new_zeros(())
                if anchor is not None:
                    selected = anchor.mask[:, 0]
                    target = anchor.labels[selected]
                    selected_logits = clean_logits.movedim(1, -1)[selected]
                    anchor_loss = F.cross_entropy(selected_logits, target)
                    anchor_accuracy = (
                        (selected_logits.argmax(dim=1) == target)
                        .to(torch.float32)
                        .mean()
                    )
                total = adversarial + self.cfg.anchor.loss_weight * anchor_loss
            self.scaler.scale(total).backward()
            self.scaler.step(self.denoiser_optimizer)
            self.scaler.update()
        finally:
            for critic in self.critics.values():
                critic.requires_grad_(True)
        return (
            float(adversarial.detach()),
            float(total.detach()),
            float(anchor_loss.detach()),
            float(anchor_accuracy.detach()),
        )

    def _sample_anchor(self) -> AnchorCondition | None:
        if not self.cfg.anchor.enabled:
            return None
        if not bool(torch.rand((), device=self.device) < self.cfg.anchor.probability):
            return None
        axis = int(torch.randint(len(AXES), (), device=self.device).item())
        labels = (
            self.streams[axis]
            .next()
            .to(
                self.device,
                non_blocking=True,
            )
        )
        if labels.ndim != 3 or labels.shape[-2:] != (
            self.cfg.data.patch_size,
            self.cfg.data.patch_size,
        ):
            raise ValueError(f"axis {axis} anchor batch must have shape [B, H, W].")
        indices = torch.randint(
            labels.shape[0],
            (self.cfg.train.volume_batch_size,),
            device=self.device,
        )
        selected = labels.index_select(0, indices)
        plane_index = int(
            torch.randint(
                self.cfg.data.patch_size,
                (),
                device=self.device,
            ).item()
        )
        return build_anchors(
            (
                PlaneAnchor(
                    labels=selected,
                    axis=axis,
                    index=plane_index,
                ),
            ),
            batch_size=self.cfg.train.volume_batch_size,
            num_phases=self.cfg.data.num_phases,
            volume_size=self.cfg.data.patch_size,
            device=self.device,
            dtype=torch.float32,
        )

    def _denoise(
        self,
        current: torch.Tensor,
        times: torch.Tensor,
        latent: torch.Tensor,
        anchor: AnchorCondition | None,
    ) -> torch.Tensor:
        if anchor is None:
            return self.denoiser(current, times, latent)
        return self.denoiser(
            current,
            times,
            latent,
            anchor_image=anchor.image,
            anchor_mask=anchor.mask,
        )

    def _predict_logits(
        self,
        current: torch.Tensor,
        times: torch.Tensor,
        latent: torch.Tensor,
        anchor: AnchorCondition | None,
    ) -> torch.Tensor:
        if anchor is None:
            return self.denoiser.predict_logits(current, times, latent)
        return self.denoiser.predict_logits(
            current,
            times,
            latent,
            anchor_image=anchor.image,
            anchor_mask=anchor.mask,
        )

    def _make_times(self, transition: int, batch: int) -> torch.Tensor:
        return torch.full(
            (batch,),
            transition,
            device=self.device,
            dtype=torch.long,
        )

    def _sample_latent(self, batch: int, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            batch,
            self.cfg.model.latent_channels,
            device=self.device,
            dtype=dtype,
        )

    def _autocast(
        self,
        *,
        enabled: bool | None = None,
    ) -> AbstractContextManager:
        if enabled is None:
            enabled = self.amp_enabled
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=enabled,
        )
