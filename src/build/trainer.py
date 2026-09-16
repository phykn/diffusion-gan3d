from pathlib import Path

import torch
from torch import nn

from src.build.data import (
    build_augmentation,
    build_datasets,
    build_stream,
    resolve_height_metadata,
)
from src.build.model import build_diffusion, build_models
from src.config import (
    get_schedule_steps,
    get_sizes,
    normalize_train_config,
)
from src.storage import load_model
from src.train.ema import build_ema
from src.train.state import fingerprint_data
from src.train.trainer import Trainer, TrainerComponents, TrainerSettings


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    connectivity_critic: nn.Module,
    cfg: dict,
) -> tuple[
    torch.optim.Optimizer,
    dict[str, torch.optim.Optimizer],
    torch.optim.Optimizer,
]:
    cfg = normalize_train_config(cfg)
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
    connectivity_optim = torch.optim.Adam(
        connectivity_critic.parameters(),
        lr=optim["critic_lr"],
        betas=betas,
    )
    return denoiser_optim, critic_optims, connectivity_optim


def build_trainer(cfg: dict, device: torch.device) -> Trainer:
    cfg = normalize_train_config(cfg)
    train = cfg["train"]
    resolve_height_metadata(cfg)
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    loss = cfg["loss"]
    conditioning = cfg["conditioning"]
    anchor = conditioning["anchor"]
    connectivity = loss["connectivity"]
    optim = cfg["optim"]
    anchor_start_step, anchor_ramp_steps = get_schedule_steps(
        anchor,
        "conditioning.anchor",
    )
    connectivity_start, connectivity_ramp = get_schedule_steps(
        connectivity, "loss.connectivity"
    )
    if (
        anchor["probability"] > 0.0
        and anchor_start_step < train["total_steps"]
        and train["volume_batch_size"] > train["real_batch_size"]
    ):
        raise ValueError(
            "train.volume_batch_size must not exceed train.real_batch_size when "
            "anchor training is enabled."
        )
    critic_augment = build_augmentation(cfg)
    denoiser, critics, connectivity_critic = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
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
    use_amp = train["mixed_precision"] and device.type == "cuda"
    datasets = build_datasets(cfg)
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

    trainer = Trainer(
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
            scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
            device=device,
            critic_augment=critic_augment,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            num_phases=data["num_phases"],
            patch_size=get_sizes(data)[1],
            slice_pairs_per_axis=train["slice_pairs_per_plane"],
            ema_decay=optim["ema_decay"],
            r1_gamma=loss["r1_weight"],
            r1_interval=loss["r1_every_steps"],
            critic_local_weight=loss["critic_local_weight"],
            anchor_training_probability=anchor["probability"],
            anchor_start_step=anchor_start_step,
            anchor_ramp_steps=anchor_ramp_steps,
            anchor_pixel_loss_weight=loss["anchor_pixel_weight"],
            anchor_shared_axis_probability=anchor["borrowed_plane_probability"],
            anchor_bank_capacity=anchor["bank_capacity"],
            anchor_plane_spacing=anchor["plane_spacing"],
            structure_every_steps=train["structure_every_steps"],
            connectivity_weight=connectivity["adversarial_weight"],
            normal_transition_weight=connectivity["normal_transition_weight"],
            connectivity_max_gap=connectivity.get("max_slice_gap", 1),
            connectivity_start_step=connectivity_start,
            connectivity_ramp_steps=connectivity_ramp,
            connectivity_windows_per_plane=connectivity["windows_per_plane"],
            vf_loss_weight=loss["volume_fraction_weight"],
            domain_dropout=1.0 - conditioning["domain_keep_probability"],
            cfg_drop_each_probability=conditioning["dropout_probability_per_case"],
            latent_channels=generator["latent_channels"],
            amp_enabled=use_amp,
            r2_gamma=loss["r2_weight"],
        ),
    )
    trainer.cfg = cfg
    trainer.height_data = cfg["data"] if cfg["conditioning"]["height_enabled"] else None
    trainer.data_fingerprint = fingerprint_data(streams)
    return trainer
