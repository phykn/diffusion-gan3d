from collections.abc import Iterable
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
        *,
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
        max_positions = len(AXES) * ((patch_size + 1) // 2)
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
        self._target_count = 0
        self._target_mean = torch.zeros(num_phases, dtype=torch.float64)
        self._target_m2 = torch.zeros(num_phases, dtype=torch.float64)

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
        done = 0
        weights = root / WEIGHTS_NAME
        print(
            f"Training Diffusion GAN3D steps=0->{steps} device={self.device} run={root}"
        )
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
                self._write_metrics(writer, done, metrics)
                bar.set_postfix(
                    G=f"{metrics.generator:.4g}",
                    D=f"{metrics.critic:.4g}",
                    t=metrics.transition,
                    A=metrics.anchor_planes,
                )
                if done % save_every == 0:
                    weights = save_training_weights(
                        root,
                        self.ema_denoiser,
                        self.critics,
                    )
            if done % save_every:
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
            print(f"Training interrupted after step {done}; weights={weights}")
            raise
        finally:
            bar.close()
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
        batches = self._next_real_batches()
        target_vf = _get_vf(
            batches.values(),
            self.num_phases,
        ).expand(self.volume_batch_size, -1)
        vf = self._apply_vf_dropout(target_vf)
        anchor = self._sample_anchor(batches)
        (
            prev,
            current,
            logits,
            clean_probs,
        ) = self._generate_pair(
            transition,
            anchor,
            vf,
        )
        fake = {
            axis: _sample_pairs(
                prev,
                current,
                axis=axis,
                count=self.slices_per_axis,
            )
            for axis in AXES
        }

        (
            critic_vals,
            r1,
            critic_global,
            critic_local,
        ) = self._update_critics(
            transition,
            fake,
            batches,
            step,
        )
        (
            g_loss,
            generator_total,
            generator_global,
            generator_local,
            anchor_loss,
            anchor_accuracy,
            vf_loss,
        ) = self._update_denoiser(
            transition,
            fake,
            logits,
            clean_probs,
            anchor,
            target_vf,
            vf_active=vf is not None,
        )
        target_vals, soft_vals, hard_vals, hard_mae = self._summarize_vfs(
            clean_probs,
            target_vf,
        )
        target_stds = self._update_target_statistics(target_vals)
        update_ema(self.ema_denoiser, self.denoiser, self.ema_decay)
        return Metrics(
            generator=g_loss,
            generator_total=generator_total,
            critic=sum(critic_vals),
            r1=r1,
            transition=transition,
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
            target_vfs=target_vals,
            target_vf_stds=target_stds,
            soft_vfs=soft_vals,
            hard_vfs=hard_vals,
            hard_vf_mae=hard_mae,
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
        writer.add_scalar("loss/vf", metrics.vf_loss, step)
        writer.add_scalar("train/transition", metrics.transition, step)
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

    def _generate_pair(
        self,
        transition: int,
        anchor: AnchorCondition | None,
        vf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                time = self._make_times(index, current.shape[0])
                latent = self._sample_latent(current.shape[0], dtype=current.dtype)
                pred = self._denoise(
                    current,
                    time,
                    latent,
                    anchor,
                    vf,
                )
                current = self.diffusion.sample_posterior(
                    current,
                    pred,
                    index,
                )

        current = current.detach()
        time = self._make_times(transition, current.shape[0])
        latent = self._sample_latent(current.shape[0], dtype=current.dtype)
        with self._autocast():
            logits = self._predict_logits(
                current,
                time,
                latent,
                anchor,
                vf,
            )
            pred = self.denoiser.decode(logits)
            previous = self.diffusion.sample_posterior(
                current,
                pred,
                transition,
            )
        clean_probs = (pred + 1.0) * 0.5
        return previous, current, logits, clean_probs

    def _update_critics(
        self,
        transition: int,
        fake: dict[int, tuple[torch.Tensor, torch.Tensor]],
        batches: dict[int, torch.Tensor],
        step: int,
    ) -> tuple[list[float], float, float, float]:
        apply_r1 = self.r1_gamma > 0.0 and (step + 1) % self.r1_interval == 0
        critic_vals = []
        r1_sum = 0.0
        global_sum = 0.0
        local_sum = 0.0
        local_w = self.critic_local_weight
        for axis in AXES:
            critic = self.critics[str(axis)]
            optim = self.critic_optims[str(axis)]
            optim.zero_grad(set_to_none=True)

            imgs = batches[axis]
            real = (
                F.one_hot(imgs, num_classes=self.num_phases)
                .movedim(-1, 1)
                .to(torch.float32)
                .mul_(2.0)
                .sub_(1.0)
            )
            real_time = self._make_times(transition, real.shape[0])
            real_prev, real_curr = self.diffusion.sample_pair(
                real,
                transition,
            )
            real_prev.requires_grad_(apply_r1)
            fake_prev, fake_curr = fake[axis]
            fake_prev = fake_prev.detach().float()
            fake_curr = fake_curr.detach().float()
            fake_time = self._make_times(transition, fake_prev.shape[0])

            autocast = self._autocast(enabled=self.amp_enabled and not apply_r1)
            with autocast:
                real_score = critic(real_prev, real_curr, real_time)
                fake_score = critic(fake_prev, fake_curr, fake_time)
                head = critic_logistic_loss(real_score, fake_score)
                loss = head.total(local_w)
            global_sum += float(head.global_loss.detach())
            local_sum += float(head.local_loss.detach())
            if apply_r1:
                r1 = critic_r1_penalty(
                    real_score,
                    (real_prev,),
                )
                penalty = r1.total(local_w)
                r1_sum += float(penalty.detach())
                loss = loss + 0.5 * self.r1_gamma * self.r1_interval * penalty
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            critic_vals.append(float(loss.detach()))
        self.scaler.update()
        return critic_vals, r1_sum, global_sum, local_sum

    def _update_denoiser(
        self,
        transition: int,
        fake: dict[int, tuple[torch.Tensor, torch.Tensor]],
        logits: torch.Tensor,
        clean_probs: torch.Tensor,
        anchor: AnchorCondition | None,
        target_vf: torch.Tensor,
        *,
        vf_active: bool,
    ) -> tuple[float, float, float, float, float, float, float]:
        self.denoiser_optim.zero_grad(set_to_none=True)
        for critic in self.critics.values():
            critic.requires_grad_(False)
        try:
            heads = []
            local_w = self.critic_local_weight
            with self._autocast():
                for axis in AXES:
                    fake_prev, fake_curr = fake[axis]
                    time = self._make_times(transition, fake_prev.shape[0])
                    scores = self.critics[str(axis)](
                        fake_prev,
                        fake_curr,
                        time,
                    )
                    heads.append(generator_logistic_loss(scores))
                global_loss = torch.stack([loss.global_loss for loss in heads]).sum()
                local_loss = torch.stack([loss.local_loss for loss in heads]).sum()
                adv = global_loss + local_w * local_loss
                anchor_loss = adv.new_zeros(())
                anchor_accuracy = adv.new_zeros(())
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
                vf_loss = adv.new_zeros(())
                if vf_active:
                    pred_vf = clean_probs.mean(dim=(2, 3, 4))
                    vf_loss = (pred_vf - target_vf).abs().sum(dim=1).mean()
                total = (
                    adv
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
            float(adv.detach()),
            float(total.detach()),
            float(global_loss.detach()),
            float(local_loss.detach()),
            float(anchor_loss.detach()),
            float(anchor_accuracy.detach()),
            float(vf_loss.detach()),
        )

    def _sample_anchor(
        self,
        batches: dict[int, torch.Tensor],
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
        positions = _sample_anchor_positions(
            count,
            volume_size=self.patch_size,
        )
        planes = []
        for axis, plane_idx in positions:
            imgs = batches[axis]
            batch_idx = torch.randint(
                imgs.shape[0],
                (self.volume_batch_size,),
                device=self.device,
            )
            selected = imgs.index_select(0, batch_idx)
            planes.append(
                PlaneAnchor(
                    image=selected,
                    axis=axis,
                    index=plane_idx,
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
        time: torch.Tensor,
        latent: torch.Tensor,
        anchor: AnchorCondition | None,
        vf: torch.Tensor | None,
    ) -> torch.Tensor:
        if anchor is None and vf is None:
            return self.denoiser(current, time, latent)
        cond = {}
        if anchor is not None:
            cond["anchor_image"] = anchor.image
            cond["anchor_mask"] = anchor.mask
        if vf is not None:
            cond["vf"] = vf
        return self.denoiser(current, time, latent, **cond)

    def _predict_logits(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        anchor: AnchorCondition | None,
        vf: torch.Tensor | None,
    ) -> torch.Tensor:
        if anchor is None and vf is None:
            return self.denoiser.predict_logits(current, time, latent)
        cond = {}
        if anchor is not None:
            cond["anchor_image"] = anchor.image
            cond["anchor_mask"] = anchor.mask
        if vf is not None:
            cond["vf"] = vf
        return self.denoiser.predict_logits(current, time, latent, **cond)

    def _next_real_batches(self) -> dict[int, torch.Tensor]:
        return {
            axis: self.streams[axis]
            .next()
            .to(
                self.device,
                non_blocking=True,
            )
            for axis in AXES
        }

    def _apply_vf_dropout(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        dropout = self.vf_dropout
        if dropout <= 0.0:
            return target
        if dropout >= 1.0 or bool(torch.rand(()) < dropout):
            return None
        return target

    def _update_target_statistics(
        self,
        target: tuple[float, ...],
    ) -> tuple[float, ...]:
        vals = torch.tensor(target, dtype=torch.float64)
        self._target_count += 1
        delta = vals - self._target_mean
        self._target_mean.add_(delta / self._target_count)
        self._target_m2.add_(delta * (vals - self._target_mean))
        if self._target_count < 2:
            return tuple(0.0 for _ in target)
        std = (self._target_m2 / (self._target_count - 1)).sqrt()
        return tuple(float(value) for value in std)

    @staticmethod
    def _summarize_vfs(
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        float,
    ]:
        vals = probs.detach().to(torch.float32)
        target_vals = target.detach().to(torch.float32).mean(dim=0)
        soft_vals = vals.mean(dim=(0, 2, 3, 4))
        phases = vals.argmax(dim=1)
        hard_vals = torch.bincount(
            phases.flatten(),
            minlength=vals.shape[1],
        ).to(torch.float32)
        hard_vals.div_(phases.numel())
        hard_mae = (hard_vals - target_vals).abs().mean()
        return (
            tuple(float(value) for value in target_vals),
            tuple(float(value) for value in soft_vals),
            tuple(float(value) for value in hard_vals),
            float(hard_mae),
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


def _get_vf(
    batches: Iterable[torch.Tensor],
    num_phases: int,
) -> torch.Tensor:
    counts = torch.stack(
        [torch.bincount(imgs.flatten(), minlength=num_phases) for imgs in batches]
    ).sum(dim=0)
    vf = counts.to(torch.float32)
    vf.div_(counts.sum())
    return vf.unsqueeze(0)


def _sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    axis: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous.shape != current.shape:
        raise ValueError("previous and current volumes must have the same shape.")
    if previous.ndim != 5:
        raise ValueError("volumes must have shape [B, C, D, H, W].")
    if axis not in AXES:
        raise ValueError("axis must be 0, 1, or 2.")
    if count <= 0:
        raise ValueError("count must be positive.")

    batch_idx = torch.randint(
        previous.shape[0],
        (count,),
        device=previous.device,
    )
    plane_idx = torch.randint(
        previous.shape[axis + 2],
        (count,),
        device=previous.device,
    )
    previous = previous.movedim(axis + 2, 2)
    current = current.movedim(axis + 2, 2)
    return previous[batch_idx, :, plane_idx], current[batch_idx, :, plane_idx]


def _sample_anchor_positions(
    count: int,
    *,
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
    for flat_idx in torch.randperm(available).tolist():
        axis, plane_idx = divmod(flat_idx, volume_size)
        if any(
            axis == prev_axis and abs(plane_idx - prev_idx) < gap
            for prev_axis, prev_idx in selected
        ):
            continue
        selected.append((axis, plane_idx))
        if len(selected) == count:
            return tuple(selected)
    raise RuntimeError("could not sample sufficiently separated anchor positions.")
