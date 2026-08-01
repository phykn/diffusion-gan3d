from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .. import AXES
from ..anchor import AnchorCondition, PlaneAnchor, build_anchors
from ..dataset import BatchStream
from ..diffusion import Diffusion
from ..model.denoiser import Denoiser3D
from .ema import update_ema
from .loss import (
    get_critic_loss,
    get_critic_r1,
    get_generator_loss,
)
from .weights import MODEL_FILE, save_all_weights


@dataclass(frozen=True)
class Metrics:
    generator: float
    generator_total: float
    critic: float
    r1: float
    transition: int
    volume_size: int
    critic_axes: tuple[float, float, float]
    anchor_planes: int
    anchor_conflict_rate: float
    anchor_loss: float
    anchor_accuracy: float
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


class Trainer:
    def __init__(
        self,
        denoiser: Denoiser3D,
        ema_denoiser: Denoiser3D,
        critics: nn.ModuleDict,
        streams: dict[int, BatchStream],
        diffusion: Diffusion,
        denoiser_optim: torch.optim.Optimizer,
        critic_optims: dict[str, torch.optim.Optimizer],
        scaler: torch.amp.GradScaler,
        device: torch.device,
        volume_batch_size: int,
        volume_sizes: Sequence[int],
        num_phases: int,
        patch_size: int,
        slices_per_axis: int,
        ema_decay: float,
        r1_gamma: float,
        r1_interval: int,
        critic_local_weight: float,
        anchor_dropout: float,
        anchor_loss_weight: float,
        anchor_max_planes: int,
        vf_loss_weight: float,
        vf_dropout: float,
        latent_channels: int,
        amp_enabled: bool,
    ) -> None:
        if set(streams) != set(AXES):
            raise ValueError("streams must contain axes 0, 1, and 2.")
        if set(critic_optims) != {str(axis) for axis in AXES}:
            raise ValueError("critic optimizers must contain axes 0, 1, and 2.")
        if not volume_sizes:
            raise ValueError("volume_sizes must not be empty.")
        max_positions = len(AXES) * ((min(volume_sizes) + 1) // 2)
        if anchor_max_planes > max_positions:
            raise ValueError(
                "anchor_max_planes exceeds the available separated positions."
            )
        self.denoiser = denoiser
        self.ema_denoiser = ema_denoiser
        self.critics = critics
        self.streams = streams
        self.diffusion = diffusion
        self.denoiser_optim = denoiser_optim
        self.critic_optims = critic_optims
        self.scaler = scaler
        self.device = device
        self.volume_batch_size = volume_batch_size
        self.volume_sizes = tuple(volume_sizes)
        self.num_phases = num_phases
        self.patch_size = patch_size
        self.slices_per_axis = slices_per_axis
        self.ema_decay = ema_decay
        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval
        self.critic_local_weight = critic_local_weight
        self.anchor_dropout = anchor_dropout
        self.anchor_loss_weight = anchor_loss_weight
        self.anchor_max_planes = anchor_max_planes
        self.vf_loss_weight = vf_loss_weight
        self.vf_dropout = vf_dropout
        self.latent_channels = latent_channels
        self.amp_enabled = amp_enabled
        self.target_count = 0
        self.target_mean = torch.zeros(num_phases, dtype=torch.float64)
        self.target_m2 = torch.zeros(num_phases, dtype=torch.float64)

    def fit(
        self,
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
        done = 0
        weights = root / MODEL_FILE
        print("\nTraining")
        print("--------")
        print(f"Steps  : {steps}")
        print(f"Device : {self.device}")
        print(f"Run    : {root}")
        writer = SummaryWriter(root / "tensorboard")
        bar = tqdm(
            range(steps),
            total=steps,
            desc="Diffusion GAN3D",
            dynamic_ncols=True,
        )
        try:
            for step in bar:
                metrics = self.step(step)
                done = step + 1
                self.write_metrics(writer, done, metrics)
                bar.set_postfix(
                    G=f"{metrics.generator:.4g}",
                    D=f"{metrics.critic:.4g}",
                    t=metrics.transition,
                    S=metrics.volume_size,
                    A=metrics.anchor_planes,
                )
                if done % save_every == 0:
                    weights = save_all_weights(
                        root,
                        self.ema_denoiser,
                        self.critics,
                    )
            if done % save_every:
                weights = save_all_weights(
                    root,
                    self.ema_denoiser,
                    self.critics,
                )
        except KeyboardInterrupt:
            weights = save_all_weights(
                root,
                self.ema_denoiser,
                self.critics,
            )
            print(f"Training interrupted after step {done}; weights={weights}")
            raise
        finally:
            bar.close()
            writer.close()
        return weights

    def step(
        self,
        step: int,
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
        volume_size = self.sample_volume_size()
        batches = self.get_batches()
        real = {
            axis: self.crop_images(images, self.patch_size)
            for axis, images in batches.items()
        }
        target_vf = self.get_vf(batches.values()).expand(self.volume_batch_size, -1)
        vf = self.apply_vf_dropout(target_vf)
        anchor = self.sample_anchor(batches, volume_size)
        (
            previous,
            current,
            logits,
            clean_probs,
        ) = self.generate_pair(
            transition,
            anchor,
            vf,
            volume_size,
        )
        fake = {
            axis: self.sample_pairs(
                previous,
                current,
                axis,
                axis_masks=None if anchor is None else anchor.axis_masks,
            )
            for axis in AXES
        }

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
        )
        (
            generator_loss,
            generator_total,
            generator_global,
            generator_local,
            anchor_loss,
            anchor_accuracy,
            vf_loss,
        ) = self.update_denoiser(
            transition,
            fake,
            logits,
            clean_probs,
            anchor,
            target_vf,
            vf_active=vf is not None,
        )
        target_values, soft_values, hard_values, hard_mae = self.summarize_vfs(
            clean_probs,
            target_vf,
        )
        target_stds = self.update_target_stats(target_values)
        update_ema(self.ema_denoiser, self.denoiser, self.ema_decay)
        return Metrics(
            generator=generator_loss,
            generator_total=generator_total,
            critic=sum(critic_vals),
            r1=r1,
            transition=transition,
            volume_size=volume_size,
            critic_axes=tuple(critic_vals),
            anchor_planes=0 if anchor is None else anchor.planes,
            anchor_conflict_rate=0.0 if anchor is None else anchor.conflict_rate,
            anchor_loss=anchor_loss,
            anchor_accuracy=anchor_accuracy,
            generator_global=generator_global,
            generator_local=generator_local,
            critic_global=critic_global,
            critic_local=critic_local,
            vf_loss=vf_loss,
            vf_active=vf is not None,
            target_vfs=target_values,
            target_vf_stds=target_stds,
            soft_vfs=soft_values,
            hard_vfs=hard_values,
            hard_vf_mae=hard_mae,
        )

    def sample_volume_size(self) -> int:
        index = int(torch.randint(len(self.volume_sizes), ()).item())
        return self.volume_sizes[index]

    def get_batches(self) -> dict[int, torch.Tensor]:
        return {
            axis: self.streams[axis]
            .next()
            .to(
                self.device,
                non_blocking=True,
            )
            for axis in AXES
        }

    @staticmethod
    def crop_images(
        images: torch.Tensor,
        size: int,
        centers: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        if images.ndim not in (3, 4):
            raise ValueError("images must have shape [B, H, W] or [B, C, H, W].")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("crop size must be a positive integer.")
        height, width = images.shape[-2:]
        if size > min(height, width):
            raise ValueError("crop size must fit inside the images.")
        if (height, width) == (size, size):
            return images

        top = torch.randint(height - size + 1, (images.shape[0],)).tolist()
        left = torch.randint(width - size + 1, (images.shape[0],)).tolist()
        if centers is not None:
            if len(centers) > images.shape[0]:
                raise ValueError("centers must not outnumber images.")
            for index, (row, col) in enumerate(centers):
                top[index] = min(max(row - size // 2, 0), height - size)
                left[index] = min(max(col - size // 2, 0), width - size)
        return torch.stack(
            [
                image[..., row : row + size, col : col + size]
                for image, row, col in zip(images, top, left, strict=True)
            ]
        )

    def get_vf(self, batches: Iterable[torch.Tensor]) -> torch.Tensor:
        counts = torch.stack(
            [
                torch.bincount(images.flatten(), minlength=self.num_phases)
                for images in batches
            ]
        ).sum(dim=0)
        vf = counts.to(torch.float32)
        vf.div_(counts.sum())
        return vf.unsqueeze(0)

    def apply_vf_dropout(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        dropout = self.vf_dropout
        if dropout <= 0.0:
            return target
        if dropout >= 1.0 or bool(torch.rand(()) < dropout):
            return None
        return target

    def sample_anchor(
        self,
        batches: dict[int, torch.Tensor],
        volume_size: int,
    ) -> AnchorCondition | None:
        if self.anchor_dropout >= 1.0:
            return None
        if not bool(torch.rand(()) < 1.0 - self.anchor_dropout):
            return None
        count = int(
            torch.randint(
                1,
                self.anchor_max_planes + 1,
                (),
            ).item()
        )
        positions = self.sample_anchor_positions(
            count,
            volume_size=volume_size,
        )
        planes = []
        for axis, plane_index in positions:
            images = batches[axis]
            batch_indices = torch.randint(
                images.shape[0],
                (self.volume_batch_size,),
                device=self.device,
            )
            selected = images.index_select(0, batch_indices)
            size = min(volume_size, *selected.shape[-2:])
            selected = self.crop_images(selected, size)
            position = tuple(
                int(torch.randint(volume_size - size + 1, ()).item()) for _ in range(2)
            )
            planes.append(
                PlaneAnchor(
                    image=selected,
                    axis=axis,
                    index=plane_index,
                    position=position,
                )
            )
        return build_anchors(
            planes,
            batch_size=self.volume_batch_size,
            num_phases=self.num_phases,
            volume_size=volume_size,
            device=self.device,
            dtype=torch.float32,
            reconcile=True,
        )

    @staticmethod
    def sample_anchor_positions(
        count: int,
        volume_size: int,
    ) -> tuple[tuple[int, int], ...]:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer.")
        if not isinstance(volume_size, int) or volume_size < 1:
            raise ValueError("volume_size must be a positive integer.")
        available = len(AXES) * volume_size
        separated = len(AXES) * ((volume_size + 1) // 2)
        if count > separated:
            raise ValueError("count exceeds the available separated positions.")

        gap = max(2, volume_size // (2 * count))
        selected: list[tuple[int, int]] = []
        for flat_index in torch.randperm(available).tolist():
            axis, plane_index = divmod(flat_index, volume_size)
            if any(
                axis == previous_axis and abs(plane_index - previous_index) < gap
                for previous_axis, previous_index in selected
            ):
                continue
            selected.append((axis, plane_index))
            if len(selected) == count:
                return tuple(selected)
        raise RuntimeError("could not sample sufficiently separated anchor positions.")

    def generate_pair(
        self,
        transition: int,
        anchor: AnchorCondition | None,
        vf: torch.Tensor | None,
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
        conditions = {}
        if anchor is not None:
            conditions["anchor_image"] = anchor.image
            conditions["anchor_mask"] = anchor.mask
        if vf is not None:
            conditions["vf"] = vf

        with torch.no_grad(), self.autocast():
            for index in reversed(range(transition + 1, self.diffusion.timesteps)):
                time = self.make_time(index, current.shape[0])
                latent = self.sample_latent(current.shape[0], current.dtype)
                prediction = self.denoiser(
                    current,
                    time,
                    latent,
                    **conditions,
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
                **conditions,
            )
            prediction = self.denoiser.decode(logits)
            previous = self.diffusion.sample_posterior(
                current,
                prediction,
                transition,
            )
        clean_probs = (prediction + 1.0) * 0.5
        return previous, current, logits, clean_probs

    def sample_pairs(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        axis: int,
        axis_masks: torch.Tensor | None = None,
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

        count = self.slices_per_axis
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
        pairs = self.crop_images(
            torch.cat((previous, current), dim=1),
            self.patch_size,
            centers,
        )
        return pairs[:, :channels], pairs[:, channels:]

    def update_critics(
        self,
        transition: int,
        fake: dict[int, tuple[torch.Tensor, torch.Tensor]],
        batches: dict[int, torch.Tensor],
        step: int,
    ) -> tuple[list[float], float, float, float]:
        apply_r1 = self.r1_gamma > 0.0 and (step + 1) % self.r1_interval == 0
        critic_losses = []
        r1_sum = 0.0
        global_sum = 0.0
        local_sum = 0.0
        local_weight = self.critic_local_weight
        for axis in AXES:
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
            real_prev.requires_grad_(apply_r1)
            fake_prev, fake_curr = fake[axis]
            fake_prev = fake_prev.detach().float()
            fake_curr = fake_curr.detach().float()
            fake_time = self.make_time(transition, fake_prev.shape[0])

            autocast = self.autocast(self.amp_enabled and not apply_r1)
            with autocast:
                real_score = critic(real_prev, real_curr, real_time)
                fake_score = critic(fake_prev, fake_curr, fake_time)
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
            critic_losses.append(float(loss.detach()))
        self.scaler.update()
        return critic_losses, r1_sum, global_sum, local_sum

    def update_denoiser(
        self,
        transition: int,
        fake: dict[int, tuple[torch.Tensor, torch.Tensor]],
        logits: torch.Tensor,
        clean_probs: torch.Tensor,
        anchor: AnchorCondition | None,
        target_vf: torch.Tensor,
        vf_active: bool,
    ) -> tuple[float, float, float, float, float, float, float]:
        self.denoiser_optim.zero_grad(set_to_none=True)
        for critic in self.critics.values():
            critic.requires_grad_(False)
        try:
            heads = []
            local_weight = self.critic_local_weight
            with self.autocast():
                for axis in AXES:
                    fake_prev, fake_curr = fake[axis]
                    time = self.make_time(transition, fake_prev.shape[0])
                    scores = self.critics[str(axis)](
                        fake_prev,
                        fake_curr,
                        time,
                    )
                    heads.append(get_generator_loss(scores))
                global_loss = torch.stack([loss.global_loss for loss in heads]).sum()
                local_loss = torch.stack([loss.local_loss for loss in heads]).sum()
                adversarial_loss = global_loss + local_weight * local_loss
                anchor_loss = adversarial_loss.new_zeros(())
                anchor_accuracy = adversarial_loss.new_zeros(())
                if anchor is not None:
                    selected = anchor.mask[:, 0]
                    anchor_target = anchor.target[selected]
                    anchor_logits = logits.movedim(1, -1)[selected]
                    anchor_loss = F.cross_entropy(anchor_logits, anchor_target)
                    anchor_accuracy = (
                        (anchor_logits.argmax(dim=1) == anchor_target)
                        .to(torch.float32)
                        .mean()
                    )
                vf_loss = adversarial_loss.new_zeros(())
                if vf_active:
                    pred_vf = clean_probs.mean(dim=(2, 3, 4))
                    vf_loss = (pred_vf - target_vf).abs().sum(dim=1).mean()
                total = (
                    adversarial_loss
                    + self.anchor_loss_weight * anchor_loss
                    + self.vf_loss_weight * vf_loss
                )
            self.scaler.scale(total).backward()
            self.scaler.step(self.denoiser_optim)
            self.scaler.update()
        finally:
            for critic in self.critics.values():
                critic.requires_grad_(True)
        return (
            float(adversarial_loss.detach()),
            float(total.detach()),
            float(global_loss.detach()),
            float(local_loss.detach()),
            float(anchor_loss.detach()),
            float(anchor_accuracy.detach()),
            float(vf_loss.detach()),
        )

    @staticmethod
    def summarize_vfs(
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        float,
    ]:
        values = probs.detach().to(torch.float32)
        target_values = target.detach().to(torch.float32).mean(dim=0)
        soft_values = values.mean(dim=(0, 2, 3, 4))
        phases = values.argmax(dim=1)
        hard_values = torch.bincount(
            phases.flatten(),
            minlength=values.shape[1],
        ).to(torch.float32)
        hard_values.div_(phases.numel())
        hard_mae = (hard_values - target_values).abs().mean()
        return (
            tuple(float(value) for value in target_values),
            tuple(float(value) for value in soft_values),
            tuple(float(value) for value in hard_values),
            float(hard_mae),
        )

    def update_target_stats(
        self,
        target: tuple[float, ...],
    ) -> tuple[float, ...]:
        values = torch.tensor(target, dtype=torch.float64)
        self.target_count += 1
        delta = values - self.target_mean
        self.target_mean.add_(delta / self.target_count)
        self.target_m2.add_(delta * (values - self.target_mean))
        if self.target_count < 2:
            return tuple(0.0 for _ in target)
        std = (self.target_m2 / (self.target_count - 1)).sqrt()
        return tuple(float(value) for value in std)

    @staticmethod
    def write_metrics(
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
        writer.add_scalar("loss/vf", metrics.vf_loss, step)
        writer.add_scalar("train/transition", metrics.transition, step)
        writer.add_scalar("train/volume_size", metrics.volume_size, step)
        writer.add_scalar("conditioning/anchor_planes", metrics.anchor_planes, step)
        writer.add_scalar(
            "conditioning/vf_active",
            float(metrics.vf_active),
            step,
        )
        writer.add_scalar(
            "conditioning/vf_hard_mae",
            metrics.hard_vf_mae,
            step,
        )
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
        for phase, vals in enumerate(
            zip(
                metrics.target_vfs,
                metrics.target_vf_stds,
                metrics.soft_vfs,
                metrics.hard_vfs,
                strict=True,
            )
        ):
            target, target_std, soft, hard = vals
            writer.add_scalar(
                f"conditioning/vf_target_{phase}",
                target,
                step,
            )
            writer.add_scalar(
                f"conditioning/vf_target_std_{phase}",
                target_std,
                step,
            )
            writer.add_scalar(
                f"conditioning/vf_soft_{phase}",
                soft,
                step,
            )
            writer.add_scalar(
                f"conditioning/vf_hard_{phase}",
                hard,
                step,
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
