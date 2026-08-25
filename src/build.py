from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

from . import AXES
from .config import (
    find_train_config,
    get_domains,
    get_schedule_steps,
    load_generation_settings,
)
from .dataset.augment import CriticAugment
from .dataset.build import build_datasets, build_stream
from .engine import Trainer, TrainerComponents, TrainerSettings
from .model.critic import ConnectivityCritic2D, PairCritic2D
from .model.denoiser import Denoiser3D
from .model.diffusion import Diffusion
from .model.ema import build_ema
from .model.generator import Generator
from .utils import load_model, load_yaml


def get_generator_channels(model: Mapping[str, object]) -> tuple[int, tuple[int, ...]]:
    generator = model["generator"]
    if not isinstance(generator, Mapping):
        raise TypeError("model.generator must be a mapping.")
    channels = generator["channels"]
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise TypeError("model.generator.channels must be a sequence.")
    values = tuple(channels)
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in values
    ):
        raise ValueError("model.generator.channels must contain positive integers.")
    base = values[0]
    if any(value % base for value in values):
        raise ValueError(
            "model.generator.channels must be integer multiples of its first value."
        )
    return base, tuple(value // base for value in values)


def build_denoiser(
    cfg: dict,
    checkpointing: bool | None = None,
) -> Denoiser3D:
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    anchor = cfg["anchor"]
    num_domains = len(get_domains(data))
    checkpointing = model["grad_checkpoint"] if checkpointing is None else checkpointing
    base_channels, multipliers = get_generator_channels(model)
    return Denoiser3D(
        num_phases=data["num_phase"],
        base_channels=base_channels,
        channel_multipliers=multipliers,
        embedding_channels=generator["condition_channels"],
        latent_channels=generator["latent_channels"],
        num_domains=num_domains,
        gradient_checkpointing=checkpointing,
        anchor_multiscale=anchor["multiscale_input"],
    )


def build_models(
    cfg: dict,
) -> tuple[Denoiser3D, nn.ModuleDict, ConnectivityCritic2D]:
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    critic = model["critic"]
    domains = get_domains(data)
    num_domains = len(domains)
    denoiser = build_denoiser(cfg)
    critics = nn.ModuleDict(
        {
            str(axis): PairCritic2D(
                num_phases=data["num_phase"],
                channels=critic["channels"],
                embedding_channels=generator["condition_channels"],
                num_domains=num_domains,
                gradient_checkpointing=model["grad_checkpoint"],
            )
            for axis in AXES
            if any(axis in folders for folders in domains.values())
        }
    )
    connectivity_critic = ConnectivityCritic2D(
        num_phases=data["num_phase"],
        channels=critic["channels"],
        embedding_channels=generator["condition_channels"],
        num_domains=num_domains,
        gradient_checkpointing=model["grad_checkpoint"],
    )
    return denoiser, critics, connectivity_critic


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
    optim = cfg["optim"]
    betas = tuple(optim["adam_betas"])
    denoiser_optim = torch.optim.Adam(
        denoiser.parameters(),
        lr=optim["generator_lr"],
        betas=betas,
    )
    critic_optims = {
        str(axis): torch.optim.Adam(
            critics[str(axis)].parameters(),
            lr=optim["critic_lr"],
            betas=betas,
        )
        for axis in sorted(int(axis) for axis in critics)
    }
    connectivity_optim = torch.optim.Adam(
        connectivity_critic.parameters(),
        lr=optim["critic_lr"],
        betas=betas,
    )
    return denoiser_optim, critic_optims, connectivity_optim


def build_diffusion(cfg: dict) -> Diffusion:
    diffusion = cfg["diffusion"]
    return Diffusion(
        diffusion["steps"],
        diffusion["beta_min"],
        diffusion["beta_max"],
    )


def load_generator(
    weights: str | Path,
    device: torch.device,
) -> Generator:
    path = Path(weights).resolve()
    config = find_train_config(path)
    cfg = load_yaml(config)
    denoiser = build_denoiser(cfg, checkpointing=False).to(device)
    try:
        load_model(path, denoiser)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    denoiser.eval()
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    generation = load_generation_settings()
    use_amp = train["amp"] and device.type == "cuda"
    return Generator(
        denoiser,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=data["input_size"],
        num_phases=data["num_phase"],
        latent_channels=model["generator"]["latent_channels"],
        use_amp=use_amp,
        anchor_spread=generation.anchor_spread,
    )


def build_trainer(cfg: dict, device: torch.device) -> Trainer:
    train = cfg["train"]
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    critic = model["critic"]
    anchor = cfg["anchor"]
    connectivity = anchor["connectivity"]
    conditioning = cfg["condition_dropout"]
    vf = cfg["vf"]
    optim = cfg["optim"]
    anchor_start_step, anchor_ramp_steps = get_schedule_steps(
        anchor,
        "anchor",
    )
    if (
        anchor["train_prob"] > 0.0
        and anchor_start_step < train["steps"]
        and train["volume_batch_size"] > data["batch_size"]
    ):
        raise ValueError(
            "train.volume_batch_size must not exceed data.batch_size when "
            "anchor training is enabled."
        )
    denoiser, critics, connectivity_critic = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    connectivity_critic = connectivity_critic.to(device)
    ema = build_ema(denoiser)
    initial_weights = train.get("init_weights")
    if initial_weights is not None:
        root = Path(initial_weights)
        load_model(root / "generator.pt", denoiser)
        load_model(root / "generator.pt", ema)
        for axis, critic_model in critics.items():
            load_model(root / f"critic_{axis}.pt", critic_model)
        load_model(root / "critic_c.pt", connectivity_critic)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    critic_augment = CriticAugment(
        data.get("augment", False),
        prob=data["augment_prob"],
    )
    use_amp = train["amp"] and device.type == "cuda"
    datasets = build_datasets(cfg)
    streams = {
        domain_id: {
            axis: build_stream(
                dataset,
                batch_size=data["batch_size"],
                num_workers=data["num_workers"],
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
            scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
            device=device,
            critic_augment=critic_augment,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            num_phases=data["num_phase"],
            patch_size=data["input_size"],
            slice_pairs_per_axis=train["pairs_per_axis"],
            ema_decay=optim["ema_decay"],
            r1_gamma=critic["r1_weight"],
            r1_interval=critic["r1_interval"],
            critic_local_weight=critic["local_loss_weight"],
            anchor_training_probability=anchor["train_prob"],
            anchor_start_step=anchor_start_step,
            anchor_ramp_steps=anchor_ramp_steps,
            anchor_pixel_loss_weight=anchor["pixel_weight"],
            anchor_shared_axis_probability=anchor["cross_domain_prob"],
            connectivity_weight=connectivity["weight"],
            normal_transition_weight=connectivity["phase_transition_weight"],
            connectivity_max_gap=connectivity.get("max_gap", 1),
            vf_loss_weight=vf["weight"],
            domain_dropout=1.0 - data["domain_prob"],
            cfg_drop_each_probability=conditioning["joint_each_prob"],
            latent_channels=generator["latent_channels"],
            amp_enabled=use_amp,
        ),
    )
