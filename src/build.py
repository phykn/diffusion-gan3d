import torch
from torch import nn

from .data import AXES, LabelPatchDataset, build_batch_stream, load_axis_paths
from .model import Denoiser3D, PairCritic2D
from .train.config import DataConfig, ModelConfig, OptimConfig


def build_axis_streams(
    cfg: DataConfig,
    *,
    device: torch.device,
):
    grouped = load_axis_paths(cfg.folder)
    return {
        axis: build_batch_stream(
            LabelPatchDataset(
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
    denoiser = Denoiser3D(
        num_phases=data.num_phases,
        base_channels=model.base_channels,
        channel_multipliers=model.channel_multipliers,
        embedding_channels=model.embedding_channels,
        latent_channels=model.latent_channels,
        gradient_checkpointing=model.gradient_checkpointing,
    )
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
