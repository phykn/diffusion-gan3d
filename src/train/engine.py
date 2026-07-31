from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..anchor import AnchorCondition, PlaneAnchor, build_anchors
from ..data.dataset import AXES, BatchStream
from ..data.slices import encode_labels, sample_pairs
from ..diffusion import Diffusion
from ..model.denoiser import Denoiser3D
from .ema import update_ema
from .loss import (
    critic_logistic_loss,
    critic_r1_penalty,
    generator_logistic_loss,
)
from .weights import WEIGHTS_NAME, save_training_weights


@dataclass(frozen=True)
class Metrics:
    generator: float
    generator_total: float
    critic: float
    r1: float
    transition: int
    critic_axes: tuple[float, float, float]
    anchor_planes: int
    anchor_conflict_rate: float
    anchor_loss: float
    anchor_accuracy: float
    generator_global: float = 0.0
    generator_local: float = 0.0
    critic_global: float = 0.0
    critic_local: float = 0.0


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
        device: torch.device,
        volume_batch_size: int,
        num_phases: int,
        patch_size: int,
        slices_per_axis: int,
        ema_decay: float,
        r1_gamma: float,
        r1_interval: int,
        critic_local_weight: float,
        anchor_probability: float,
        anchor_loss_weight: float,
        anchor_max_planes: int,
        latent_channels: int,
        amp_enabled: bool,
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
        self.device = device
        self.volume_batch_size = volume_batch_size
        self.num_phases = num_phases
        self.patch_size = patch_size
        self.slices_per_axis = slices_per_axis
        self.ema_decay = ema_decay
        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval
        self.critic_local_weight = critic_local_weight
        self.anchor_probability = anchor_probability
        self.anchor_loss_weight = anchor_loss_weight
        self.anchor_max_planes = anchor_max_planes
        self.latent_channels = latent_channels
        self.amp_enabled = amp_enabled

    def fit(
        self,
        *,
        steps: int,
        save_every: int,
        run_dir: str | Path,
    ) -> Path:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer.")
        if (
            not isinstance(save_every, int)
            or isinstance(save_every, bool)
            or save_every < 1
        ):
            raise ValueError("save_every must be a positive integer.")
        root = Path(run_dir)
        completed = 0
        weights = root / WEIGHTS_NAME
        print(
            f"Training Diffusion GAN3D steps=0->{steps} device={self.device} run={root}"
        )
        writer = SummaryWriter(root / "tensorboard")
        progress = tqdm(
            range(steps),
            total=steps,
            desc="Diffusion GAN3D",
            dynamic_ncols=True,
        )
        try:
            for step in progress:
                metrics = self.step(step)
                completed = step + 1
                self._write_metrics(writer, completed, metrics)
                progress.set_postfix(
                    G=f"{metrics.generator:.4g}",
                    D=f"{metrics.critic:.4g}",
                    t=metrics.transition,
                    A=metrics.anchor_planes,
                )
                if completed % save_every == 0:
                    weights = save_training_weights(
                        root,
                        self.ema_denoiser,
                        self.critics,
                    )
            if completed % save_every:
                weights = save_training_weights(
                    root,
                    self.ema_denoiser,
                    self.critics,
                )
        except KeyboardInterrupt:
            weights = save_training_weights(
                root,
                self.ema_denoiser,
                self.critics,
            )
            print(f"Training interrupted after step {completed}; weights={weights}")
            raise
        finally:
            progress.close()
            writer.close()
        return weights

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
                count=self.slices_per_axis,
            )
            for axis in AXES
        }

        (
            critic_values,
            r1_value,
            critic_global,
            critic_local,
        ) = self._update_critics(
            transition,
            fake_pairs,
            step,
        )
        (
            generator_value,
            generator_total,
            generator_global,
            generator_local,
            anchor_loss,
            anchor_accuracy,
        ) = self._update_denoiser(
            transition,
            fake_pairs,
            clean_logits,
            anchor,
        )
        update_ema(self.ema_denoiser, self.denoiser, self.ema_decay)
        return Metrics(
            generator=generator_value,
            generator_total=generator_total,
            critic=sum(critic_values),
            r1=r1_value,
            transition=transition,
            critic_axes=tuple(critic_values),
            anchor_planes=0 if anchor is None else anchor.planes,
            anchor_conflict_rate=0.0 if anchor is None else anchor.conflict_rate,
            anchor_loss=anchor_loss,
            anchor_accuracy=anchor_accuracy,
            generator_global=generator_global,
            generator_local=generator_local,
            critic_global=critic_global,
            critic_local=critic_local,
        )

    @staticmethod
    def _write_metrics(
        writer: SummaryWriter,
        step: int,
        metrics: Metrics,
    ) -> None:
        writer.add_scalar("loss/generator", metrics.generator, step)
        writer.add_scalar("loss/generator_total", metrics.generator_total, step)
        writer.add_scalar("loss/generator_global", metrics.generator_global, step)
        writer.add_scalar("loss/generator_local_raw", metrics.generator_local, step)
        writer.add_scalar("loss/critic_total", metrics.critic, step)
        writer.add_scalar("loss/critic_global", metrics.critic_global, step)
        writer.add_scalar("loss/critic_local_raw", metrics.critic_local, step)
        writer.add_scalar("loss/r1_raw", metrics.r1, step)
        writer.add_scalar("train/transition", metrics.transition, step)
        writer.add_scalar("conditioning/anchor_planes", metrics.anchor_planes, step)
        if metrics.anchor_planes:
            writer.add_scalar("loss/anchor", metrics.anchor_loss, step)
            writer.add_scalar(
                "conditioning/anchor_accuracy",
                metrics.anchor_accuracy,
                step,
            )
            writer.add_scalar(
                "conditioning/anchor_conflict_rate",
                metrics.anchor_conflict_rate,
                step,
            )
            writer.add_scalar(
                f"loss/anchor_{metrics.anchor_planes}_planes",
                metrics.anchor_loss,
                step,
            )
            writer.add_scalar(
                f"conditioning/anchor_accuracy_{metrics.anchor_planes}_planes",
                metrics.anchor_accuracy,
                step,
            )
        for axis, value in zip(AXES, metrics.critic_axes, strict=True):
            writer.add_scalar(f"loss/critic_axis_{axis}", value, step)

    def _generate_pair(
        self,
        transition: int,
        anchor: AnchorCondition | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (
            self.volume_batch_size,
            self.num_phases,
            self.patch_size,
            self.patch_size,
            self.patch_size,
        )
        current = torch.randn(shape, device=self.device, dtype=torch.float32)
        with torch.no_grad(), self._autocast():
            for index in reversed(range(transition + 1, self.diffusion.timesteps)):
                times = self._make_times(index, current.shape[0])
                latent = self._sample_latent(current.shape[0], dtype=current.dtype)
                clean = self._denoise(current, times, latent, anchor)
                if anchor is not None:
                    clean = anchor.project(clean)
                current = self.diffusion.sample_posterior(
                    current,
                    clean,
                    index,
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
            if anchor is not None:
                clean = anchor.project(clean)
            previous = self.diffusion.sample_posterior(
                current,
                clean,
                transition,
            )
        return previous, current, clean_logits

    def _update_critics(
        self,
        transition: int,
        fake_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> tuple[list[float], float, float, float]:
        apply_r1 = self.r1_gamma > 0.0 and (step + 1) % self.r1_interval == 0
        critic_values = []
        total_r1 = 0.0
        total_global = 0.0
        total_local = 0.0
        local_weight = self.critic_local_weight
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
            real_clean = encode_labels(labels, self.num_phases)
            real_times = self._make_times(transition, real_clean.shape[0])
            real_previous, real_current = self.diffusion.sample_pair(
                real_clean,
                transition,
            )
            real_previous.requires_grad_(apply_r1)
            fake_previous, fake_current = fake_pairs[axis]
            fake_previous = fake_previous.detach().float()
            fake_current = fake_current.detach().float()
            fake_times = self._make_times(transition, fake_previous.shape[0])

            autocast = self._autocast(enabled=self.amp_enabled and not apply_r1)
            with autocast:
                real_scores = critic(real_previous, real_current, real_times)
                fake_scores = critic(fake_previous, fake_current, fake_times)
                head_loss = critic_logistic_loss(real_scores, fake_scores)
                loss = head_loss.total(local_weight)
            total_global += float(head_loss.global_loss.detach())
            total_local += float(head_loss.local_loss.detach())
            if apply_r1:
                r1 = critic_r1_penalty(
                    real_scores,
                    (real_previous,),
                )
                penalty = r1.total(local_weight)
                total_r1 += float(penalty.detach())
                loss = loss + 0.5 * self.r1_gamma * self.r1_interval * penalty
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            critic_values.append(float(loss.detach()))
        self.scaler.update()
        return critic_values, total_r1, total_global, total_local

    def _update_denoiser(
        self,
        transition: int,
        fake_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]],
        clean_logits: torch.Tensor,
        anchor: AnchorCondition | None,
    ) -> tuple[float, float, float, float, float, float]:
        self.denoiser_optimizer.zero_grad(set_to_none=True)
        for critic in self.critics.values():
            critic.requires_grad_(False)
        try:
            head_losses = []
            local_weight = self.critic_local_weight
            with self._autocast():
                for axis in AXES:
                    fake_previous, fake_current = fake_pairs[axis]
                    times = self._make_times(transition, fake_previous.shape[0])
                    scores = self.critics[str(axis)](
                        fake_previous,
                        fake_current,
                        times,
                    )
                    head_losses.append(generator_logistic_loss(scores))
                global_loss = torch.stack(
                    [loss.global_loss for loss in head_losses]
                ).sum()
                local_loss = torch.stack(
                    [loss.local_loss for loss in head_losses]
                ).sum()
                adversarial = global_loss + local_weight * local_loss
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
                total = adversarial + self.anchor_loss_weight * anchor_loss
            self.scaler.scale(total).backward()
            self.scaler.step(self.denoiser_optimizer)
            self.scaler.update()
        finally:
            for critic in self.critics.values():
                critic.requires_grad_(True)
        return (
            float(adversarial.detach()),
            float(total.detach()),
            float(global_loss.detach()),
            float(local_loss.detach()),
            float(anchor_loss.detach()),
            float(anchor_accuracy.detach()),
        )

    def _sample_anchor(self) -> AnchorCondition | None:
        if self.anchor_probability <= 0.0:
            return None
        if not bool(torch.rand(()) < self.anchor_probability):
            return None
        count = int(
            torch.randint(
                1,
                self.anchor_max_planes + 1,
                (),
            ).item()
        )
        # Random precedence prevents one axis from owning every reconciled line.
        axes = torch.randperm(len(AXES))[:count].tolist()
        planes = []
        for axis in axes:
            labels = (
                self.streams[axis]
                .next()
                .to(
                    self.device,
                    non_blocking=True,
                )
            )
            indices = torch.randint(
                labels.shape[0],
                (self.volume_batch_size,),
                device=self.device,
            )
            selected = labels.index_select(0, indices)
            plane_index = int(
                torch.randint(
                    self.patch_size,
                    (),
                ).item()
            )
            planes.append(
                PlaneAnchor(
                    labels=selected,
                    axis=axis,
                    index=plane_index,
                )
            )
        return build_anchors(
            planes,
            batch_size=self.volume_batch_size,
            num_phases=self.num_phases,
            volume_size=self.patch_size,
            device=self.device,
            dtype=torch.float32,
            reconcile=True,
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
            self.latent_channels,
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
