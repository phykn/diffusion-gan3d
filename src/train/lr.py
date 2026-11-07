"""LR stage: conditions from real crops and the LR generator objective."""

import torch

from src.config.schema import LRConfig
from src.data.height import image_profile, rebin_profile
from src.data.pairs import sample_pairs
from src.stats.labels import phase_profile
from src.train.core import StepInputs, Trainer
from src.train.losses.group_statistics import group_statistics_loss
from src.train.losses.mirror_balance import mirror_balance_loss
from src.train.losses.spatial_profile import profile_loss
from src.train.losses.transition import real_transition_loss
from src.train.losses.volume_fraction import vf_loss


def schedule_ramp(step: int, start: int, ramp: int) -> float:
    if step < start:
        return 0.0
    if ramp == 0:
        return 1.0
    return min((step - start + 1) / ramp, 1.0)


class LRTrainer(Trainer):
    config: LRConfig

    def prepare(self, step: int, transition: int | None) -> StepInputs:
        config = self.config
        conditioning = config.conditioning
        height = conditioning.height_enabled
        profile_enabled = conditioning.spatial_profile.enabled
        batch = config.train.volume_batch_size
        real = self.real_crops(side_heights=height)
        # Height, profile and vf of a volume share one side crop.
        side = next((axis for axis in self.axes if axis != 0), None) if height else None
        if side is None:
            axis = self.axes[
                int(torch.randint(len(self.axes), (), generator=self.random["train"]))
            ]
        else:
            axis = side
        fractions = real[axis].images.float().mean((2, 3))
        indices = torch.arange(batch, device=fractions.device) % len(fractions)
        target_vf = fractions[indices]
        # Samples that keep vf and profile; the rest train the CFG branch.
        vf_present = (
            torch.rand(batch, generator=self.random["train"])
            >= conditioning.cfg_dropout_probability
        )
        conditions = {"vf": target_vf, "vf_present": vf_present.to(self.device)}
        extra = {}
        if height:
            conditions["height"] = self.volume_height(
                real[side].origins[:batch], real[side].extents[:batch]
            )
        if profile_enabled:
            profile = rebin_profile(
                image_profile(
                    real[side].images, bins=conditioning.spatial_profile.num_bins
                ),
                self.patch_size,
            )[:batch]
            target_vf = profile.mean(-1)
            conditions.update(
                profile=profile,
                profile_present=vf_present.to(self.device),
                vf=target_vf,
            )
            extra["profile"] = profile
        extra.update(
            target_vf=target_vf,
            vf_present=vf_present,
            real_transition_ramp=schedule_ramp(
                step,
                config.loss.real_transition.start_step,
                config.loss.real_transition.ramp_steps,
            ),
            group_statistics_ramp=schedule_ramp(
                step,
                config.loss.group_statistics.start_step,
                config.loss.group_statistics.ramp_steps,
            ),
        )
        return StepInputs(
            transition=self.sample_transition() if transition is None else transition,
            real=real,
            conditions=conditions,
            extra=extra,
        )

    def objective(self, inputs, clean_probs, fake, real, diagnose):
        loss = self.config.loss
        extra = inputs.extra
        diagnostics = {}
        vf = vf_loss(clean_probs, extra["target_vf"], extra["vf_present"])
        diagnostics["loss/vf"] = vf.detach()
        total = loss.volume_fraction_weight * vf
        real_transition = loss.real_transition
        if real_transition.weight > 0 and inputs.transition == 0:
            value = self.real_transition(inputs, clean_probs, diagnostics)
            diagnostics["loss/real_transition"] = value.detach()
            ramp = extra["real_transition_ramp"]
            total = total + ramp * real_transition.weight * value
        group_statistics = loss.group_statistics
        ramp = extra["group_statistics_ramp"]
        if group_statistics.weight > 0 and inputs.transition == 0 and ramp > 0:
            value, values = group_statistics_loss(
                clean_probs,
                self.groups,
                group_statistics.max_gap,
                group_statistics.tolerance,
                self.config.conditioning.height_enabled,
            )
            total = total + group_statistics.weight * ramp * value
            diagnostics.update(values)
            diagnostics["loss/group_statistics"] = value.detach()
            diagnostics["group_statistics/ramp"] = value.new_tensor(ramp)
        mirror_balance = loss.mirror_balance
        if mirror_balance.weight > 0:
            # The x0 prediction of every transition must keep the mirror symmetry.
            value, values = mirror_balance_loss(
                clean_probs, mirror_balance.max_gap, mirror_balance.tolerance
            )
            total = total + mirror_balance.weight * value
            diagnostics.update(values)
            diagnostics["loss/mirror_balance"] = value.detach()
        if "profile" in extra:
            total = total + self.profile(clean_probs, extra["profile"], diagnostics)
        return total, diagnostics

    def real_transition(self, inputs, clean_probs, diagnostics) -> torch.Tensor:
        real, fake, real_heights, fake_heights = {}, {}, {}, {}
        height = inputs.conditions.get("height")
        for axis in self.axes:
            crops = inputs.real[axis]
            if height is not None and crops.heights is None:
                continue
            real[axis] = crops.images
            # Sample before critic augmentation so directions keep their meaning.
            sampled = sample_pairs(
                clean_probs,
                clean_probs,
                axis,
                self.config.train.slice_pairs_per_plane,
                tuple(crops.images.shape[-2:]),
                self.random["pairs"],
                height=height,
            )
            fake[axis] = sampled[0]
            if height is not None:
                real_heights[axis] = crops.heights
                fake_heights[axis] = sampled[2]
        if not fake:
            return clean_probs.sum(dtype=torch.float32) * 0.0
        value, values = real_transition_loss(
            real,
            fake,
            self.config.loss.real_transition.max_gap,
            real_heights if height is not None else None,
            fake_heights if height is not None else None,
            self.augment.rotated_axes,
        )
        diagnostics.update(values)
        return value

    def profile(self, clean_probs, target, diagnostics) -> torch.Tensor:
        # The real crop profile is the target for CFG-dropped samples too: the
        # height field alone must carry the depth gradient at generation.
        config = self.config
        value = profile_loss(
            clean_probs,
            target,
            torch.ones(len(target), dtype=torch.bool, device=target.device),
            config.conditioning.spatial_profile.num_bins,
            config.loss.spatial_profile_weight,
        )
        detached = clean_probs.detach()
        soft = detached.mean((-1, -2))
        labels = phase_profile(detached.argmax(1), self.num_phases)
        diagnostics.update(
            {
                "profile/loss": value.detach(),
                "profile/soft_mae": (soft - target).abs().mean(),
                "profile/label_mae": (labels - target).abs().mean(),
            }
        )
        return value

    def stage_metrics(self, inputs: StepInputs) -> dict:
        present = inputs.extra["vf_present"]
        return {
            "conditioning/vf_active": present.any(),
            "conditioning/vf_fraction": present.float().mean(),
            "loss/real_transition_ramp": inputs.extra["real_transition_ramp"],
        }
