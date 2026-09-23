from dataclasses import dataclass

import torch

from src.evaluate.profile import phase_profile
from src.train.batch import DenoiserBatch, DenoiserUpdate
from src.train.loss.connectivity import compute_transition_loss
from src.train.loss.gan import (
    HeadLoss,
    active_groups,
    get_generator_loss,
    score_plane,
)
from src.train.loss.spatial_profile import compute_profile_loss
from src.train.loss.sr import consistency_loss
from src.train.loss.volume_fraction import compute_vf_loss
from src.train.step import input_gradient_norms


@dataclass(frozen=True)
class DenoiserLossSettings:
    local_weight: float
    connectivity_weight: float
    normal_transition_weight: float
    vf_weight: float
    real_transition_weight: float
    profile_bins: int
    profile_weight: float
    profile_gradient_weight: float
    consistency_weight: float
    consistency_tolerance: float
    num_phases: int


def denoiser_objective(
    batch: DenoiserBatch,
    critics,
    connectivity_critic,
    groups: dict[str, tuple[int, ...]],
    anchor_loss,
    settings: DenoiserLossSettings,
    diagnose: bool = False,
) -> tuple[DenoiserUpdate, dict[str, torch.Tensor]]:
    """Compute the differentiable objective without updating training state."""
    head, diagnostics = adversarial_loss(
        batch, critics, groups, settings.local_weight, diagnose
    )
    global_loss, local_loss = head.global_loss, head.local_loss
    adversarial = head.combine(settings.local_weight)
    connectivity = adversarial.new_zeros(())
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
        connectivity = get_generator_loss(scores).combine(settings.local_weight)
    normal = adversarial.new_zeros(())
    if len(batch.connectivity_fake) and settings.normal_transition_weight > 0.0:
        normal = compute_transition_loss(
            batch.connectivity_real, batch.connectivity_fake
        )
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
        + batch.connectivity_ramp
        * (
            settings.connectivity_weight * connectivity
            + settings.normal_transition_weight * normal
        )
        + settings.vf_weight * vf
    )
    if batch.real_transition_loss is not None:
        total = total + (
            batch.connectivity_ramp
            * settings.real_transition_weight
            * batch.real_transition_loss
        )
        diagnostics["loss/real_transition"] = batch.real_transition_loss.detach()
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
        global_loss=global_loss,
        local_loss=local_loss,
        connectivity=connectivity,
        normal_transition=normal,
        anchor=anchor,
        anchor_coarse=anchor_coarse,
        anchor_pixel=anchor_pixel,
        anchor_accuracy=anchor_accuracy,
        vf=vf,
    ), diagnostics


def adversarial_loss(batch, critics, groups, local_weight, diagnose):
    heads = []
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
        head = get_generator_loss(scores)
        if diagnose:
            # Undo only the batch mean, retaining local and pyramid weights.
            norms = input_gradient_norms(
                head.combine(local_weight) * len(fake_prev), (fake_prev, fake_curr)
            )
            prefix = f"generator_input_gradient/{axis_critics[axis]}/{axis}/t{batch.transition}"
            for name, norm in zip(("previous", "current"), norms):
                diagnostics[f"{prefix}/{name}"] = norm
        count = len(groups[axis_critics[axis]]) * len(groups)
        heads.append(HeadLoss(head.global_loss / count, head.local_loss / count))
    global_loss = (
        torch.stack([loss.global_loss for loss in heads]).sum()
        if heads
        else batch.logits.sum() * 0
    )
    local_loss = (
        torch.stack([loss.local_loss for loss in heads]).sum()
        if heads
        else batch.logits.sum() * 0
    )
    return HeadLoss(global_loss, local_loss), diagnostics


def profile_loss(batch, settings):
    loss = compute_profile_loss(
        batch.clean_probs,
        batch.profile,
        batch.vf_present,
        settings.profile_bins,
        settings.profile_gradient_weight,
        settings.profile_weight,
    )
    soft = batch.clean_probs.detach().mean((-1, -2))
    labels = phase_profile(batch.clean_probs.detach().argmax(1), settings.num_phases)
    active = batch.vf_present.to(soft).reshape(-1, 1, 1)
    denom = active.sum().clamp_min(1) * soft.shape[1] * soft.shape[2]
    return loss, {
        "profile/loss": loss.detach(),
        "profile/soft_mae": ((soft - batch.profile).abs() * active).sum() / denom,
        "profile/label_mae": ((labels - batch.profile).abs() * active).sum() / denom,
    }


def coarse_loss(batch, settings):
    consistency, error = consistency_loss(
        batch.clean_probs, batch.coarse_target, settings.consistency_tolerance
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
