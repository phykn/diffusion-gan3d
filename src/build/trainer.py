from pathlib import Path

import torch
from torch import nn

from src.build.data import (
    build_augmentation,
    build_datasets,
    build_stream,
)
from src.build.model import build_diffusion, build_models
from src.config.data import get_sizes, get_sr_plane_groups
from src.config.train import get_schedule_steps, get_sr_sizes, normalize_train_config
from src.data.source import infer_height_extents
from src.storage import load_model
from src.train.ema import build_ema
from src.train.state import fingerprint_data
from src.train.trainer import Trainer, TrainerComponents, TrainerSettings


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    connectivity_critic: nn.Module | None,
    cfg: dict,
) -> tuple[
    torch.optim.Optimizer,
    dict[str, torch.optim.Optimizer],
    torch.optim.Optimizer | None,
]:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    optim = cfg["optim"]
    betas = tuple(optim["adam_betas"])
    denoiser_optim = torch.optim.Adam(
        denoiser.parameters(),
        lr=optim["generator_lr"],
        betas=betas,
    )
    critic_optims = {
        plane: torch.optim.Adam(
            critics[plane].parameters(),
            lr=optim["critic_lr"],
            betas=betas,
        )
        for plane in critics
    }
    connectivity_optim = (
        None
        if connectivity_critic is None
        else torch.optim.Adam(
            connectivity_critic.parameters(),
            lr=optim["critic_lr"],
            betas=betas,
        )
    )
    return denoiser_optim, critic_optims, connectivity_optim


def build_trainer(
    cfg: dict, device: torch.device, bank=None, bank_origins=None, bank_extents=None
) -> Trainer:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    sr = cfg["stage"] == "sr"
    if sr != (bank is not None):
        raise ValueError("only SR training requires a coarse bank.")
    train = cfg["train"]
    data = cfg["data"]
    if cfg["conditioning"]["height_enabled"]:
        data["height_extents"] = infer_height_extents(data)
    settings = _build_settings(cfg, device)
    critic_augment = build_augmentation(cfg)
    denoiser, critics, connectivity_critic = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    if connectivity_critic is not None:
        connectivity_critic = connectivity_critic.to(device)
    ema = build_ema(denoiser)
    initial_weights = train.get("initial_weights")
    if initial_weights is not None:
        root = Path(initial_weights)
        load_model(root / "generator.pt", denoiser)
        load_model(root / "generator.pt", ema)
        for plane, critic_model in critics.items():
            path = root / f"critic_{plane}.pt"
            load_model(path, critic_model)
        load_model(root / "critic_c.pt", connectivity_critic)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    datasets = build_datasets(cfg, high=sr)
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
            scaler=torch.amp.GradScaler("cuda", enabled=settings.amp_enabled),
            device=device,
            critic_augment=critic_augment,
            coarse_bank=bank,
            bank_origins=bank_origins,
            bank_extents=bank_extents,
            data_fingerprint=fingerprint_data(streams),
            critic_groups_by_domain=get_sr_plane_groups(cfg) if sr else None,
        ),
        settings=settings,
    )


def _build_settings(cfg: dict, device: torch.device) -> TrainerSettings:
    sr = cfg["stage"] == "sr"
    train = cfg["train"]
    data = cfg["data"]
    generator = cfg["model"]["generator"]
    loss = cfg["loss"]
    conditioning = cfg["conditioning"]
    anchor = conditioning.get("anchor", {})
    connectivity = loss.get("connectivity", {})
    optim = cfg["optim"]
    anchor_start_step, anchor_ramp_steps = (
        (0, 0) if sr else get_schedule_steps(anchor, "conditioning.anchor")
    )
    connectivity_start, connectivity_ramp = (
        (0, 0) if sr else get_schedule_steps(connectivity, "loss.connectivity")
    )
    if (
        anchor.get("probability", 0.0) > 0.0
        and anchor_start_step < train["total_steps"]
        and train["volume_batch_size"] > train["real_batch_size"]
    ):
        raise ValueError(
            "train.volume_batch_size must not exceed train.real_batch_size when "
            "anchor training is enabled."
        )
    if (
        conditioning["height_enabled"]
        and train["volume_batch_size"] > train["real_batch_size"]
    ):
        raise ValueError(
            "height conditioning requires real_batch_size >= volume_batch_size."
        )
    return TrainerSettings(
        cfg=cfg,
        height_data=data if conditioning["height_enabled"] else None,
        profile_settings=conditioning.get(
            "spatial_profile", {"enabled": False, "num_bins": 16}
        ),
        profile_weight=loss.get("spatial_profile_weight", 0.0),
        profile_gradient_weight=loss.get("spatial_profile_gradient_weight", 0.0),
        volume_batch_size=train["volume_batch_size"],
        num_phases=data["num_phases"],
        patch_size=get_sr_sizes(cfg)[2] if sr else get_sizes(data)[1],
        slice_pairs_per_axis=train["slice_pairs_per_plane"],
        ema_decay=optim["ema_decay"],
        r1_gamma=loss["r1_weight"],
        r1_interval=loss["r1_every_steps"],
        critic_local_weight=loss["critic_local_weight"],
        anchor_training_probability=anchor.get("probability", 0.0),
        anchor_start_step=anchor_start_step,
        anchor_ramp_steps=anchor_ramp_steps,
        anchor_pixel_loss_weight=loss.get("anchor_pixel_weight", 0.0),
        anchor_shared_axis_probability=anchor.get("borrowed_plane_probability", 0.0),
        anchor_bank_capacity=anchor.get("bank_capacity", 0),
        anchor_plane_spacing=anchor.get("plane_spacing", 1),
        structure_every_steps=train["structure_every_steps"],
        connectivity_weight=connectivity.get("adversarial_weight", 0.0),
        normal_transition_weight=connectivity.get("normal_transition_weight", 0.0),
        connectivity_max_gap=connectivity.get("max_slice_gap", 1),
        connectivity_start_step=connectivity_start,
        connectivity_ramp_steps=connectivity_ramp,
        connectivity_windows_per_plane=connectivity.get("windows_per_plane", 1),
        vf_loss_weight=loss.get("volume_fraction_weight", 0.0),
        domain_dropout=1.0 - conditioning["domain_keep_probability"],
        cfg_drop_each_probability=conditioning.get("dropout_probability_per_case", 0.0),
        latent_channels=generator["latent_channels"],
        amp_enabled=train["mixed_precision"] and device.type == "cuda",
        r2_gamma=loss["r2_weight"],
        consistency_weight=loss.get("downsample_consistency_weight", 0.0),
        consistency_tolerance=loss.get("downsample_mse_tolerance", 0.0),
        coarse_corruption_probability=conditioning.get(
            "coarse_corruption_probability", 0.0
        ),
        coarse_corruption_strength=conditioning.get("coarse_corruption_strength", 0.0),
    )
