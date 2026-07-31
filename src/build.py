import torch
from torch import nn

from .data import AXES, SliceDataset, build_stream, find_slices
from .diffusion import Diffusion
from .model import Denoiser3D, PairCritic2D
from .train.config import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    OptimConfig,
)


def build_streams(
    cfg: DataConfig,
    *,
    device: torch.device,
):
    grouped = find_slices(cfg.folder)
    return {
        axis: build_stream(
            SliceDataset(
                grouped[axis],
                crop_size=cfg.crop_size,
                patch_size=cfg.patch_size,
                num_phases=cfg.num_phases,
            ),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        for axis in AXES
    }


def build_models(
    data: DataConfig,
    model: ModelConfig,
) -> tuple[Denoiser3D, nn.ModuleDict]:
    denoiser = build_denoiser(data, model)
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
    data: DataConfig,
    model: ModelConfig,
    *,
    checkpointing: bool | None = None,
) -> Denoiser3D:
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


def build_diffusion(cfg: DiffusionConfig) -> Diffusion:
    return Diffusion(
        cfg.timesteps,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
    )


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    cfg: OptimConfig,
) -> tuple[torch.optim.Optimizer, dict[str, torch.optim.Optimizer]]:
    betas = (cfg.beta1, cfg.beta2)
    denoiser_optimizer = torch.optim.Adam(
        denoiser.parameters(),
        lr=cfg.generator_lr,
        betas=betas,
    )
    critic_optimizers = {
        str(axis): torch.optim.Adam(
            critics[str(axis)].parameters(),
            lr=cfg.critic_lr,
            betas=betas,
        )
        for axis in AXES
    }
    return denoiser_optimizer, critic_optimizers
