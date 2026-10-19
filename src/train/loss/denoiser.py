from dataclasses import dataclass

import torch

from src.evaluate.profile import phase_profile
from src.train.batch import DenoiserBatch, DenoiserUpdate
from src.train.loss.gan import (
    active_groups,
    get_generator_loss,
    score_plane,
)
from src.train.loss.group_statistics import compute_group_statistics_loss
from src.train.loss.spatial_profile import compute_profile_loss
from src.train.loss.sr import consistency_loss
from src.train.loss.volume_fraction import compute_vf_loss
from src.train.step import input_gradient_norms


@dataclass(frozen=True)
class DenoiserLossSettings:
    connectivity_local_weight: float
    connectivity_weight: float
    vf_weight: float
    real_transition_weight: float
    profile_bins: int
    profile_weight: float
    profile_gradient_weight: float
    consistency_weight: float
    consistency_tolerance: float
    num_phases: int
    group_statistics_weight: float = 0.0
    group_statistics_max_gap: int = 8
    group_statistics_tolerance: float = 0.01
    preserve_height: bool = False
    anchor_continuity_weight: float = 0.0
    anchor_neighborhood_weight: float = 0.0
    anchor_neighborhood_radius: int = 2
    anchor_seam_weight: float = 0.0
    anchor_seam_rows: int = 4


def denoiser_objective(
    batch: DenoiserBatch,
    critics,
    connectivity_critic,
    seam_critic,
    groups: dict[str, tuple[int, ...]],
    anchor_loss,
    settings: DenoiserLossSettings,
    diagnose: bool = False,
) -> tuple[DenoiserUpdate, dict[str, torch.Tensor]]:
    """Compute the differentiable objective without updating training state."""
    adversarial, diagnostics = adversarial_loss(batch, critics, groups, diagnose)
    connectivity = adversarial.new_zeros(())
    connectivity_global = adversarial.new_zeros(())
    connectivity_local = adversarial.new_zeros(())
    if len(batch.connectivity_fake) and settings.connectivity_weight > 0.0:
        fake = batch.connectivity_fake
        scores = connectivity_critic(
            fake.values,
            fake.axes,
            fake.gaps,
            batch.connectivity_domains,
            **({"height": fake.height} if fake.height is not None else {}),
            **({"profile": fake.profile} if fake.profile is not None else {}),
        )
        head = get_generator_loss(scores)
        connectivity_global, connectivity_local = head.global_loss, head.local_loss
        connectivity = head.combine(settings.connectivity_local_weight)
    anchor = adversarial.new_zeros(())
    anchor_coarse = adversarial.new_zeros(())
    anchor_pixel = adversarial.new_zeros(())
    anchor_accuracy = adversarial.new_zeros(())
    if batch.anchor is not None:
        result = anchor_loss(
            batch.logits,
            batch.anchor,
            batch.anchor_present,
            batch.anchor_observed_mask,
            batch.anchor_observed_axis_masks,
        )
        anchor = result.total
        anchor_coarse = result.coarse
        anchor_pixel = result.pixel
        anchor_accuracy = result.accuracy
    vf = compute_vf_loss(batch.clean_probs, batch.target_vf, batch.vf_present)
    total = (
        adversarial
        + batch.anchor_ramp * anchor
        + batch.connectivity_ramp * (settings.connectivity_weight * connectivity)
        + settings.vf_weight * vf
    )
    if batch.real_transition_loss is not None:
        total = total + (
            batch.connectivity_ramp
            * settings.real_transition_weight
            * batch.real_transition_loss
        )
        diagnostics["loss/real_transition"] = batch.real_transition_loss.detach()
    if batch.anchor_continuity_loss is not None:
        contribution = (
            batch.anchor_ramp
            * settings.anchor_continuity_weight
            * batch.anchor_continuity_loss
        )
        total = total + contribution
        diagnostics["loss/anchor_continuity"] = batch.anchor_continuity_loss.detach()
        diagnostics["anchor_continuity/contribution"] = contribution.detach()
    if batch.anchor_neighborhood_loss is not None:
        contribution = (
            batch.anchor_ramp
            * settings.anchor_neighborhood_weight
            * batch.anchor_neighborhood_loss
        )
        total = total + contribution
        diagnostics["loss/anchor_neighborhood"] = (
            batch.anchor_neighborhood_loss.detach()
        )
        diagnostics["anchor_neighborhood/contribution"] = contribution.detach()
    if (
        batch.seam_fake is not None
        and len(batch.seam_fake)
        and settings.anchor_seam_weight > 0.0
    ):
        fake = batch.seam_fake
        time = torch.full(
            (len(fake),),
            batch.transition,
            device=fake.values.device,
            dtype=torch.long,
        )
        seam = get_generator_loss(
            seam_critic(fake.values, fake.axes, time, batch.seam_domains)
        ).global_loss
        contribution = batch.anchor_ramp * settings.anchor_seam_weight * seam
        total = total + contribution
        diagnostics["loss/anchor_seam"] = seam.detach()
        diagnostics["anchor_seam/contribution"] = contribution.detach()
    if (
        settings.group_statistics_weight > 0
        and batch.transition == 0
        and batch.group_statistics_ramp > 0
    ):
        observed = (
            groups
            if batch.observed_axes is None
            else {
                name: tuple(axis for axis in axes if axis in batch.observed_axes)
                for name, axes in groups.items()
            }
        )
        active = None if batch.anchor is None else ~batch.anchor_present.bool()
        statistics, values = compute_group_statistics_loss(
            batch.clean_probs,
            observed,
            settings.group_statistics_max_gap,
            settings.group_statistics_tolerance,
            settings.preserve_height,
            active,
            batch.coarse_target,
        )
        total = total + (
            settings.group_statistics_weight * batch.group_statistics_ramp * statistics
        )
        diagnostics.update(values)
        diagnostics["loss/group_statistics"] = statistics.detach()
        diagnostics["group_statistics/ramp"] = statistics.new_tensor(
            batch.group_statistics_ramp
        )
        diagnostics["group_statistics/active_fraction"] = (
            statistics.new_tensor(1)
            if active is None
            else active.float().mean().to(statistics)
        )
    if batch.profile is not None:
        profile, values = profile_loss(batch, settings)
        total = total + profile
        diagnostics.update(values)
    if batch.coarse_target is not None:
        consistency, values = coarse_loss(batch, settings)
        total = total + settings.consistency_weight * consistency
        diagnostics.update(values)
    return DenoiserUpdate(
        adversarial=adversarial,
        total=total,
        connectivity=connectivity,
        connectivity_global=connectivity_global,
        connectivity_local=connectivity_local,
        anchor=anchor,
        anchor_coarse=anchor_coarse,
        anchor_pixel=anchor_pixel,
        anchor_accuracy=anchor_accuracy,
        vf=vf,
    ), diagnostics


def adversarial_loss(batch, critics, groups, diagnose):
    losses = []
    diagnostics = {}
    groups = active_groups(batch.fake, groups)
    axis_critics = {axis: group for group, axes in groups.items() for axis in axes}
    for axis in sorted(axis_critics):
        fake_prev, fake_curr = batch.fake[axis]
        if diagnose:
            # The preceding reverse chain is already detached during sampling.
            fake_curr = fake_curr.detach().requires_grad_(True)
        time = torch.full(
            (len(fake_prev),),
            batch.transition,
            device=fake_prev.device,
            dtype=torch.long,
        )
        domains = torch.full(
            (len(fake_prev),),
            batch.critic_domains[axis],
            device=fake_prev.device,
            dtype=torch.long,
        )
        conditions = {}
        if axis in batch.fake_heights:
            conditions["height"] = batch.fake_heights[axis]
            if axis in batch.fake_profiles:
                conditions["profile"] = batch.fake_profiles[axis]
        scores = score_plane(
            critics[axis_critics[axis]],
            fake_prev,
            fake_curr,
            time,
            domains,
            **conditions,
        )
        loss = get_generator_loss(scores).global_loss
        if diagnose:
            # Undo only the batch mean, retaining pyramid weights.
            norms = input_gradient_norms(loss * len(fake_prev), (fake_prev, fake_curr))
            prefix = f"generator_input_gradient/{axis_critics[axis]}/{axis}/t{batch.transition}"
            for name, norm in zip(("previous", "current"), norms):
                diagnostics[f"{prefix}/{name}"] = norm
        count = len(groups[axis_critics[axis]]) * len(groups)
        losses.append(loss / count)
    loss = torch.stack(losses).sum() if losses else batch.logits.sum() * 0
    return loss, diagnostics


def profile_supervised(batch, settings):
    # With height conditioning the real crop profile is still the target for
    # samples whose vf/profile was dropped for CFG: the height field alone must
    # carry the depth gradient at inference.
    if settings.preserve_height:
        return torch.ones_like(batch.vf_present, dtype=torch.bool)
    return batch.vf_present.bool()


def profile_loss(batch, settings):
    supervised = profile_supervised(batch, settings)
    loss = compute_profile_loss(
        batch.clean_probs,
        batch.profile,
        supervised,
        settings.profile_bins,
        settings.profile_gradient_weight,
        settings.profile_weight,
    )
    soft = batch.clean_probs.detach().mean((-1, -2))
    labels = phase_profile(batch.clean_probs.detach().argmax(1), settings.num_phases)
    active = supervised.to(soft).reshape(-1, 1, 1)
    denom = active.sum().clamp_min(1) * soft.shape[1] * soft.shape[2]
    return loss, {
        "profile/loss": loss.detach(),
        "profile/soft_mae": ((soft - batch.profile).abs() * active).sum() / denom,
        "profile/label_mae": ((labels - batch.profile).abs() * active).sum() / denom,
    }


def coarse_loss(batch, settings):
    consistency, error = consistency_loss(
        batch.clean_probs,
        batch.coarse_target,
        settings.consistency_tolerance,
        batch.coarse_mask,
    )
    depth = batch.coarse_target.shape[-3]
    target = phase_profile(batch.coarse_target.detach(), settings.num_phases)
    soft = phase_profile(batch.clean_probs.detach(), settings.num_phases, depth)
    labels = phase_profile(
        batch.clean_probs.detach().argmax(1), settings.num_phases, depth
    )
    return consistency, {
        "consistency": consistency.detach(),
        "lr_mse": error.detach(),
        "profile/sr_coarse_soft_mae": (soft - target).abs().mean(),
        "profile/sr_coarse_label_mae": (labels - target).abs().mean(),
    }
