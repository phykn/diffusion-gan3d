from pathlib import Path

import torch
from torch import nn

from .data.dataset import AXES, BatchStream, SliceDataset, build_stream, find_slices
from .diffusion import Diffusion
from .generate.sample import Sampler
from .model.critic import PairCritic2D
from .model.denoiser import Denoiser3D
from .simul.config import SimulationConfig
from .simul.export import Export, generate
from .train.config import TrainConfig, load_config
from .train.ema import build_ema
from .train.engine import Trainer
from .train.weights import load_weights


def build_streams(
    cfg: TrainConfig,
    *,
    device: torch.device,
) -> dict[int, BatchStream]:
    datasets = build_datasets(cfg)
    data = cfg.data
    return {
        axis: build_stream(
            datasets[axis],
            batch_size=data.batch_size,
            num_workers=data.num_workers,
            pin_memory=device.type == "cuda",
        )
        for axis in AXES
    }


def build_datasets(cfg: TrainConfig) -> dict[int, SliceDataset]:
    data = cfg.data
    grouped = find_slices(data.folder)
    return {
        axis: SliceDataset(
            grouped[axis],
            crop_size=data.crop_size,
            patch_size=data.patch_size,
            num_phases=data.num_phases,
        )
        for axis in AXES
    }


def build_models(cfg: TrainConfig) -> tuple[Denoiser3D, nn.ModuleDict]:
    data = cfg.data
    model = cfg.model
    denoiser = build_denoiser(cfg)
    critics = nn.ModuleDict(
        {
            str(axis): PairCritic2D(
                num_phases=data.num_phases,
                channels=model.critic_channels,
                embedding_channels=model.embedding_channels,
                gradient_checkpointing=model.gradient_checkpointing,
            )
            for axis in AXES
        }
    )
    return denoiser, critics


def build_denoiser(
    cfg: TrainConfig,
    *,
    checkpointing: bool | None = None,
) -> Denoiser3D:
    data = cfg.data
    model = cfg.model
    use_checkpointing = (
        model.gradient_checkpointing if checkpointing is None else checkpointing
    )
    return Denoiser3D(
        num_phases=data.num_phases,
        base_channels=model.base_channels,
        channel_multipliers=model.channel_multipliers,
        embedding_channels=model.embedding_channels,
        latent_channels=model.latent_channels,
        gradient_checkpointing=use_checkpointing,
    )


def build_diffusion(cfg: TrainConfig) -> Diffusion:
    diffusion = cfg.diffusion
    return Diffusion(
        diffusion.timesteps,
        beta_min=diffusion.beta_min,
        beta_max=diffusion.beta_max,
    )


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    cfg: TrainConfig,
) -> tuple[torch.optim.Optimizer, dict[str, torch.optim.Optimizer]]:
    optim = cfg.optim
    betas = (optim.beta1, optim.beta2)
    denoiser_optimizer = torch.optim.Adam(
        denoiser.parameters(),
        lr=optim.generator_lr,
        betas=betas,
    )
    critic_optimizers = {
        str(axis): torch.optim.Adam(
            critics[str(axis)].parameters(),
            lr=optim.critic_lr,
            betas=betas,
        )
        for axis in AXES
    }
    return denoiser_optimizer, critic_optimizers


def build_trainer(
    cfg: TrainConfig,
    *,
    device: torch.device,
) -> Trainer:
    denoiser, critics = build_models(cfg)
    _load_checkpoints(cfg, denoiser, critics)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema_denoiser = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg,
    )
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    return Trainer(
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=critics,
        streams=build_streams(cfg, device=device),
        diffusion=build_diffusion(cfg).to(device),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
        device=device,
        volume_batch_size=cfg.train.volume_batch_size,
        num_phases=cfg.data.num_phases,
        patch_size=cfg.data.patch_size,
        slices_per_axis=cfg.train.slices_per_axis,
        ema_decay=cfg.train.ema_decay,
        r1_gamma=cfg.optim.r1_gamma,
        r1_interval=cfg.optim.r1_interval,
        critic_local_weight=cfg.optim.critic_local_weight,
        anchor_probability=cfg.anchor.probability,
        anchor_loss_weight=cfg.anchor.loss_weight,
        anchor_max_planes=cfg.anchor.max_planes,
        latent_channels=cfg.model.latent_channels,
        amp_enabled=use_amp,
    )


def _load_checkpoints(
    cfg: TrainConfig,
    denoiser: nn.Module,
    critics: nn.ModuleDict,
) -> None:
    checkpoint = cfg.train.checkpoint
    if checkpoint.model is not None:
        load_weights(checkpoint.model, denoiser)
    for axis in (0, 1, 2):
        path = getattr(checkpoint, f"critic_{axis}")
        if path is not None:
            load_weights(path, critics[str(axis)])


def build_sampler(
    cfg: TrainConfig,
    model: Denoiser3D,
    *,
    device: torch.device,
    mixed_precision: bool | None = None,
) -> Sampler:
    if mixed_precision is not None and not isinstance(mixed_precision, bool):
        raise TypeError("mixed_precision must be boolean or None.")
    use_amp = (
        cfg.train.mixed_precision if mixed_precision is None else mixed_precision
    ) and device.type == "cuda"
    return Sampler(
        model,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=cfg.data.patch_size,
        num_phases=cfg.data.num_phases,
        latent_channels=cfg.model.latent_channels,
        anchor_enabled=cfg.anchor.enabled,
        max_anchor_planes=cfg.anchor.max_planes,
        use_amp=use_amp,
    )


def load_sampler(
    weights: str | Path,
    *,
    device: torch.device,
    mixed_precision: bool | None = None,
) -> Sampler:
    path = Path(weights).resolve()
    cfg = load_config(path.parent / "train.yaml")
    model = build_denoiser(cfg, checkpointing=False).to(device)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    model.eval()
    return build_sampler(
        cfg,
        model,
        device=device,
        mixed_precision=mixed_precision,
    )


def generate_data(cfg: SimulationConfig) -> Export:
    output = cfg.output
    geometry = cfg.geometry
    return generate(
        data_dir=output.data_dir,
        count=output.count,
        size=geometry.size,
        big_radius=geometry.big_radius,
        small_radius=geometry.small_radius,
        big_fraction=geometry.big_fraction,
        small_fraction=geometry.small_fraction,
        big_elongation=geometry.big_elongation,
    )
