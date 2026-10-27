import torch
from torch import nn

from src.build.data import (
    build_augmentation,
    build_datasets,
    build_stream,
)
from src.build.model import build_diffusion, build_models
from src.config.data import get_resolution, get_sr_plane_groups
from src.config.train import (
    get_schedule_steps,
    get_sr_resolution,
    get_trainer_sections,
    normalize_train_config,
    validate_training_settings,
)
from src.data.provenance import fingerprint_data
from src.data.source import fill_height_extents
from src.device import gpu_amp
from src.plane import PLANE_DIRECTIONS
from src.train.burn_in import BurnIn
from src.train.ema import build_ema
from src.train.loss.denoiser import DenoiserLossSettings
from src.train.selection import (
    CheckpointSelection,
    SRCheckpointSelection,
    build_reference,
)
from src.train.trainer import Trainer, TrainerComponents, TrainerSettings


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    cfg: dict,
) -> tuple[torch.optim.Optimizer, dict[str, torch.optim.Optimizer]]:
    optim = cfg["optim"]
    betas = tuple(optim["adam_betas"])

    def adam(module, lr):
        # Fused Adam runs one kernel per step instead of several per tensor.
        return torch.optim.Adam(module.parameters(), lr=lr, betas=betas, fused=True)

    return (
        adam(denoiser, optim["generator_lr"]),
        {plane: adam(critics[plane], optim["critic_lr"]) for plane in critics},
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
    denoiser, critics = build_models(cfg)
    denoiser, critics = denoiser.to(device), critics.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(denoiser, critics, cfg)
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
    selection = None
    if cfg["selection"]["enabled"] and sr:
        selection = SRCheckpointSelection.from_bank(
            build_reference(datasets),
            cfg["selection"]["samples"],
            bank,
            bank_origins,
            bank_extents,
        )
    elif cfg["selection"]["enabled"]:
        selection = CheckpointSelection(
            build_reference(datasets),
            cfg["selection"]["samples"],
            cfg["conditioning"]["height_enabled"],
        )

    return Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema,
            critics=critics,
            streams=streams,
            diffusion=build_diffusion(cfg).to(device),
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            scaler=torch.amp.GradScaler(device.type, enabled=settings.denoiser_amp),
            device=device,
            critic_augment=critic_augment,
            coarse_bank=bank,
            bank_origins=bank_origins,
            bank_extents=bank_extents,
            data_fingerprint=fingerprint_data(streams),
            critic_groups_by_domain=get_sr_plane_groups(cfg) if sr else None,
            selection=selection,
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
    transition = loss["real_transition"]
    statistics = loss["group_statistics"]
    mirror = loss["mirror_balance"]
    optim = cfg["optim"]
    burn_in = cfg["burn_in"]
    transition_start, transition_ramp = get_schedule_steps(
        transition, "loss.real_transition"
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
    flips = sr and cfg["lr_bank"]["flips"]
    flip_z = (
        flips
        and free_z
        and (not z_policies or any("z" in p.get("flip_axes", ()) for p in z_policies))
    )
    return TrainerSettings(
        height_data=data if conditioning["height_enabled"] else None,
        volume_permute_z=permute_z,
        volume_flip_z=flip_z,
        volume_flip_xy=flips,
        profile_settings=profile,
        loss=DenoiserLossSettings(
            real_transition_weight=transition["weight"],
            vf_weight=loss["volume_fraction_weight"],
            profile_bins=profile["num_bins"],
            profile_weight=loss["spatial_profile_weight"],
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
        slice_pairs_per_plane=train["slice_pairs_per_plane"],
        burn_in=BurnIn(
            steps=burn_in["steps"],
            generator_lr=optim["generator_lr"],
            critic_lr=optim["critic_lr"],
            betas=tuple(optim["adam_betas"]),
            ema_decay=optim["ema_decay"],
            lr_scale=burn_in["lr_scale"],
            beta2=burn_in["beta2"],
            regularization_scale=burn_in["regularization_scale"],
        ),
        r1_weight=loss["r1_weight"],
        r1_every_steps=loss["r1_every_steps"],
        structure_every_steps=train["structure_every_steps"],
        group_statistics_start_step=statistics["start_step"],
        group_statistics_ramp_steps=statistics["ramp_steps"],
        real_transition_max_gap=transition["max_gap"],
        real_transition_start_step=transition_start,
        real_transition_ramp_steps=transition_ramp,
        domain_keep_probability=conditioning["domain_keep_probability"],
        cfg_dropout_probability=conditioning["cfg_dropout_probability"],
        latent_channels=generator["latent_channels"],
        denoiser_amp=gpu_amp(train["mixed_precision"], device),
        critic_amp=train["mixed_precision"] and device.type == "cuda",
        r2_weight=loss["r2_weight"],
        coarse_corruption_probability=conditioning["coarse_corruption_probability"],
        coarse_corruption_strength=conditioning["coarse_corruption_strength"],
    )
