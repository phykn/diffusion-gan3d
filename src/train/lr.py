import torch

from src.axes import PLANES
from src.data.profile import measure_image_profile
from src.data.resize import rebin_profile
from src.stats.labels import phase_profile
from src.train.core import StepInputs, Trainer, sample_sections, schedule_ramp
from src.train.losses.spatial_profile import profile_loss
from src.train.losses.transition import real_transition_loss
from src.train.losses.volume_fraction import vf_loss


class LRTrainer(Trainer):
    def prepare(self, step, transition):
        real = self.real_crops()
        # Height, profile and vf of a volume share one side crop.
        choices = (
            tuple(axis for axis in self.axes if axis != 0) if self.height else self.axes
        )
        if self.height and len(choices) == 1:
            axis = choices[0]
        else:
            index = torch.randint(len(choices), (), generator=self.random["train"])
            axis = choices[int(index)]
        conditions, extra, diagnostics = self.crop_conditions(real[axis])
        if self.height:
            diagnostics.update(
                {
                    f"conditioning/source/{PLANES[side]}": int(axis == side)
                    for side in choices
                }
            )
        return StepInputs(
            step=step,
            transition=self.sample_transition() if transition is None else transition,
            real=real,
            conditions=conditions,
            extra=extra,
            diagnostics=diagnostics,
        )

    def crop_conditions(self, crops):
        """Denoiser conditions, loss targets and metrics from one side crop."""
        conditioning = self.config.conditioning
        batch = self.config.train.volume_batch_size
        indices = torch.arange(batch, device=crops.images.device) % len(crops.images)
        # Samples that keep vf and profile; the rest train the CFG branch.
        present = (
            torch.rand(batch, generator=self.random["train"])
            >= conditioning.cfg_dropout_probability
        )
        conditions = {"vf_present": present.to(self.device)}
        extra = {"vf_present": present}
        if self.height:
            conditions["height"] = self.volume_height(
                crops.origins[indices.cpu()], crops.extents[indices.cpu()]
            )
        if conditioning.spatial_profile.enabled:
            profile = rebin_profile(
                measure_image_profile(
                    crops.images, bins=conditioning.spatial_profile.num_bins
                ),
                self.patch_size,
            )[indices]
            target_vf = profile.mean(-1)
            conditions.update(profile=profile, profile_present=present.to(self.device))
            extra["profile"] = profile
        else:
            target_vf = crops.images.float().mean((2, 3))[indices]
        conditions["vf"] = target_vf
        extra["target_vf"] = target_vf
        diagnostics = {
            "conditioning/vf_active": present.any(),
            "conditioning/vf_fraction": present.float().mean(),
        }
        return conditions, extra, diagnostics

    def objective(self, inputs, clean_probs):
        extra = inputs.extra
        vf = vf_loss(clean_probs, extra["target_vf"], extra["vf_present"])
        total = self.config.loss.volume_fraction_weight * vf
        metrics = {"loss/vf": vf.detach()}
        for value, values in (
            self.real_transition_term(inputs, clean_probs),
            self.group_statistics_term(inputs, clean_probs),
            self.mirror_balance_term(clean_probs),
            self.profile_term(clean_probs, extra.get("profile")),
        ):
            metrics.update(values)
            if value is not None:
                total = total + value
        return total, metrics

    def real_transition_term(self, inputs, clean_probs):
        real_transition = self.config.loss.real_transition
        ramp = schedule_ramp(inputs.step, real_transition.ramp_steps)
        metrics = {"loss/real_transition_ramp": ramp}
        if not (real_transition.weight > 0 and inputs.transition == 0):
            return None, metrics
        real, fake, real_heights, fake_heights = {}, {}, {}, {}
        height = inputs.conditions.get("height")
        for axis in self.axes:
            crops = inputs.real[axis]
            if height is not None and crops.heights is None:
                continue
            real[axis] = crops.images
            # Sample before critic augmentation so directions keep their meaning.
            volumes = (clean_probs,)
            if height is not None:
                volumes += (height.detach(),)
            sampled = sample_sections(
                volumes,
                axis,
                self.config.train.slice_pairs_per_plane,
                self.random["pairs"],
            )
            fake[axis] = sampled[0]
            if height is not None:
                real_heights[axis] = crops.heights
                fake_heights[axis] = sampled[1]
        value, values = real_transition_loss(
            real,
            fake,
            real_transition.max_gap,
            real_heights if height is not None else None,
            fake_heights if height is not None else None,
            self.augment.rotatable_axes,
        )
        metrics.update(values)
        metrics["loss/real_transition"] = value.detach()
        return ramp * real_transition.weight * value, metrics

    def profile_term(self, clean_probs, target):
        if target is None:
            return None, {}
        # The real crop profile is the target for CFG-dropped samples too: the
        # height field alone must carry the depth gradient at generation.
        conditioning = self.config.conditioning
        value = profile_loss(
            clean_probs,
            target,
            conditioning.spatial_profile.num_bins,
            self.config.loss.spatial_profile_weight,
        )
        detached = clean_probs.detach()
        soft = detached.mean((-1, -2))
        labels = phase_profile(detached.argmax(1), self.num_phases)
        return value, {
            "profile/loss": value.detach(),
            "profile/soft_mae": (soft - target).abs().mean(),
            "profile/label_mae": (labels - target).abs().mean(),
        }
