from pathlib import Path

import torch
from torch import nn

from src.build.data import (
    build_augmentation,
    build_datasets,
    build_stream,
)
from src.build.model import build_diffusion, build_models, build_seam_critic
from src.config.data import get_resolution, get_sr_plane_groups
from src.config.files import PROJECT_ROOT
from src.config.train import (
    get_schedule_steps,
    get_sr_resolution,
    get_trainer_sections,
    normalize_train_config,
    validate_training_settings,
)
from src.data.provenance import fingerprint_data
from src.data.source import fill_height_extents
from src.plane import PLANE_DIRECTIONS
from src.storage import load_model
from src.train.ema import build_ema
from src.train.loss.denoiser import DenoiserLossSettings
from src.train.trainer import Trainer, TrainerComponents, TrainerSettings


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    connectivity_critic: nn.Module | None,
    seam_critic: nn.Module | None,
    cfg: dict,
) -> tuple[
    torch.optim.Optimizer,
    dict[str, torch.optim.Optimizer],
    torch.optim.Optimizer | None,
    torch.optim.Optimizer | None,
]:
    optim = cfg["optim"]
    betas = tuple(optim["adam_betas"])

    def critic_adam(module):
        if module is None:
            return None
        return torch.optim.Adam(module.parameters(), lr=optim["critic_lr"], betas=betas)

    return (
        torch.optim.Adam(denoiser.parameters(), lr=optim["generator_lr"], betas=betas),
        {plane: critic_adam(critics[plane]) for plane in critics},
        critic_adam(connectivity_critic),
        critic_adam(seam_critic),
    )


def build_trainer(
    cfg: dict, device: torch.device, bank=None, bank_origins=None, bank_extents=None
) -> Trainer:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    validate_training_settings(cfg)
    sr = cfg["stage"] == "sr"
    if sr != (bank is not None):
        raise ValueError("only SR training requires a coarse bank.")
    train = cfg["train"]
    # Loading or resuming already recorded extents; infer only for raw configs.
    fill_height_extents(cfg)
    settings = _build_settings(cfg, device)
    critic_augment = build_augmentation(cfg)
    denoiser, critics, connectivity_critic = build_models(cfg)
    seam_critic = build_seam_critic(cfg)
    initial_weights = None if sr else train["initial_weights"]
    if initial_weights is not None:
        root = PROJECT_ROOT / Path(initial_weights).expanduser()
        load_model(root / "generator.pt", denoiser)
        for plane, critic_model in critics.items():
            load_model(root / f"critic_{plane}.pt", critic_model)
        load_model(root / "critic_c.pt", connectivity_critic)
        load_model(root / "critic_s.pt", seam_critic)
    denoiser, critics = denoiser.to(device), critics.to(device)
    if connectivity_critic is not None:
        connectivity_critic = connectivity_critic.to(device)
    if seam_critic is not None:
        seam_critic = seam_critic.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim, seam_optim = build_optimizers(
        denoiser, critics, connectivity_critic, seam_critic, cfg
    )
    datasets = build_datasets(cfg, high_resolution=sr)
    streams = {
        domain_id: {
            axis: build_stream(
                dataset,
                batch_size=train["real_batch_size"],
                num_workers=train["num_workers"],
                pin_memory=device.type == "cuda",
            )
            for axis, dataset in axes.items()
        }
        for domain_id, axes in datasets.items()
    }

    return Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema,
            critics=critics,
            connectivity_critic=connectivity_critic,
            streams=streams,
            diffusion=build_diffusion(cfg).to(device),
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            connectivity_optim=connectivity_optim,
            seam_critic=seam_critic,
            seam_optim=seam_optim,
            scaler=torch.amp.GradScaler("cuda", enabled=settings.amp_enabled),
            device=device,
            critic_augment=critic_augment,
            coarse_bank=bank,
            bank_origins=bank_origins,
            bank_extents=bank_extents,
            data_fingerprint=fingerprint_data(streams),
            critic_groups_by_domain=get_sr_plane_groups(cfg) if sr else None,
            cfg=cfg,
        ),
        settings=settings,
    )


def _build_settings(cfg: dict, device: torch.device) -> TrainerSettings:
    sr = cfg["stage"] == "sr"
    train = cfg["train"]
    data = cfg["data"]
    generator = cfg["model"]["generator"]
    conditioning, loss = get_trainer_sections(cfg)
    profile = conditioning["spatial_profile"]
    anchor = conditioning["anchor"]
    connectivity = loss["connectivity"]
    statistics = loss["group_statistics"]
    mirror = loss["mirror_balance"]
    optim = cfg["optim"]
    anchor_start_step, anchor_ramp_steps = get_schedule_steps(
        anchor, "conditioning.anchor"
    )
    connectivity_start, connectivity_ramp = get_schedule_steps(
        connectivity, "loss.connectivity"
    )
    z_policies = [
        policy
        for plane, policy in (cfg["augmentation"].get("planes") or {}).items()
        if "z" in PLANE_DIRECTIONS[plane]
    ]
    # SR bank volumes swap or flip z only where the critic policies do; with no
    # z plane policy, nothing in training fixes the z orientation.
    free_z = not conditioning["height_enabled"]
    permute_z = free_z and (
        not z_policies or any(p.get("rotate_90", False) for p in z_policies)
    )
    flip_z = free_z and (
        not z_policies or any("z" in p.get("flip_axes", ()) for p in z_policies)
    )
    return TrainerSettings(
        height_data=data if conditioning["height_enabled"] else None,
        volume_permute_z=permute_z,
        volume_flip_z=flip_z,
        profile_settings=profile,
        loss=DenoiserLossSettings(
            connectivity_local_weight=connectivity["local_weight"],
            connectivity_weight=connectivity["adversarial_weight"],
            real_transition_weight=connectivity["real_transition_weight"],
            anchor_continuity_weight=loss["anchor_continuity_weight"],
            anchor_neighborhood_weight=loss["anchor_neighborhood_weight"],
            anchor_seam_weight=loss["anchor_seam_weight"],
            anchor_seam_rows=loss["anchor_seam_rows"],
            vf_weight=loss["volume_fraction_weight"],
            profile_bins=profile["num_bins"],
            profile_weight=loss["spatial_profile_weight"],
            profile_gradient_weight=loss["spatial_profile_gradient_weight"],
            consistency_weight=loss["downsample_consistency_weight"],
            consistency_tolerance=loss["downsample_mse_tolerance"],
            num_phases=data["num_phases"],
            group_statistics_weight=statistics["weight"],
            group_statistics_max_gap=statistics["max_gap"],
            group_statistics_tolerance=statistics["tolerance"],
            preserve_height=conditioning["height_enabled"],
            mirror_balance_weight=mirror["weight"],
            mirror_balance_max_gap=mirror["max_gap"],
            mirror_balance_tolerance=mirror["tolerance"],
        ),
        volume_batch_size=train["volume_batch_size"],
        num_phases=data["num_phases"],
        patch_size=get_sr_resolution(cfg).hi_res_size
        if sr
        else get_resolution(data).lo_res_size,
        slice_pairs_per_axis=train["slice_pairs_per_plane"],
        ema_decay=optim["ema_decay"],
        r1_gamma=loss["r1_weight"],
        r1_interval=loss["r1_every_steps"],
        anchor_training_probability=anchor["probability"],
        anchor_start_step=anchor_start_step,
        anchor_ramp_steps=anchor_ramp_steps,
        anchor_pixel_loss_weight=loss["anchor_pixel_weight"],
        anchor_shared_axis_probability=anchor["borrowed_plane_probability"],
        anchor_bank_capacity=anchor["bank_capacity"],
        anchor_plane_spacing=anchor["plane_spacing"],
        structure_every_steps=train["structure_every_steps"],
        group_statistics_start_step=statistics["start_step"],
        group_statistics_ramp_steps=statistics["ramp_steps"],
        connectivity_max_gap=connectivity["max_slice_gap"],
        connectivity_start_step=connectivity_start,
        connectivity_ramp_steps=connectivity_ramp,
        connectivity_windows_per_plane=connectivity["windows_per_plane"],
        domain_dropout=1.0 - conditioning["domain_keep_probability"],
        cfg_drop_each_probability=conditioning["dropout_probability_per_case"],
        latent_channels=generator["latent_channels"],
        amp_enabled=train["mixed_precision"] and device.type == "cuda",
        r2_gamma=loss["r2_weight"],
        coarse_corruption_probability=conditioning["coarse_corruption_probability"],
        coarse_corruption_strength=conditioning["coarse_corruption_strength"],
    )
