from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from . import AXES
from .dataset import BatchStream, SliceDataset
from .diffusion import Diffusion
from .generate import Generator
from .model.critic import PairCritic2D
from .model.denoiser import Denoiser3D
from .simul.config import SimulationConfig
from .simul.export import Export, generate
from .train.config import TrainConfig, load_config
from .train.ema import build_ema
from .train.engine import Trainer
from .train.weights import WEIGHTS_NAME, load_weights

_EXTENSIONS = {".png", ".tif", ".tiff"}


def find_slices(folders: dict[int, Path]) -> dict[int, tuple[Path, ...]]:
    if set(folders) != set(AXES):
        raise ValueError("axis folders must contain exactly axes 0, 1, and 2.")

    grouped = {}
    for axis in AXES:
        folder = Path(folders[axis])
        if not folder.is_dir():
            raise FileNotFoundError(f"axis {axis} folder does not exist: {folder}")
        paths = tuple(
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in _EXTENSIONS
            )
        )
        if not paths:
            raise ValueError(f"axis {axis} folder contains no images: {folder}")
        grouped[axis] = paths
    return grouped


def build_stream(
    dataset: SliceDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=max(1024, batch_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return BatchStream(loader)


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
    checkpointing = (
        model.gradient_checkpointing if checkpointing is None else checkpointing
    )
    return Denoiser3D(
        num_phases=data.num_phases,
        base_channels=model.base_channels,
        channel_multipliers=model.channel_multipliers,
        embedding_channels=model.embedding_channels,
        latent_channels=model.latent_channels,
        gradient_checkpointing=checkpointing,
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
    denoiser_optim = torch.optim.Adam(
        denoiser.parameters(),
        lr=optim.generator_lr,
        betas=betas,
    )
    critic_optims = {
        str(axis): torch.optim.Adam(
            critics[str(axis)].parameters(),
            lr=optim.critic_lr,
            betas=betas,
        )
        for axis in AXES
    }
    return denoiser_optim, critic_optims


def build_trainer(
    cfg: TrainConfig,
    *,
    device: torch.device,
) -> Trainer:
    denoiser, critics = build_models(cfg)
    _load_checkpoints(cfg, denoiser, critics)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(
        denoiser,
        critics,
        cfg,
    )
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    return Trainer(
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=build_streams(cfg, device=device),
        diffusion=build_diffusion(cfg).to(device),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
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
        anchor_dropout=cfg.anchor.dropout,
        anchor_loss_weight=cfg.anchor.loss_weight,
        anchor_max_planes=cfg.anchor.max_planes,
        vf_loss_weight=cfg.vf.loss_weight,
        vf_dropout=cfg.vf.dropout,
        latent_channels=cfg.model.latent_channels,
        amp_enabled=use_amp,
    )


def _load_checkpoints(
    cfg: TrainConfig,
    denoiser: nn.Module,
    critics: nn.ModuleDict,
) -> None:
    ckpt = cfg.train.checkpoint
    if ckpt.model is not None:
        load_weights(ckpt.model, denoiser)
    for axis in (0, 1, 2):
        path = getattr(ckpt, f"critic_{axis}")
        if path is not None:
            load_weights(path, critics[str(axis)])


def build_generator(
    cfg: TrainConfig,
    model: Denoiser3D,
    *,
    device: torch.device,
) -> Generator:
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    return Generator(
        model,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=cfg.data.patch_size,
        num_phases=cfg.data.num_phases,
        latent_channels=cfg.model.latent_channels,
        anchor_enabled=cfg.anchor.enabled,
        use_amp=use_amp,
    )


def find_weights(run_root: str | Path) -> Path:
    root = Path(run_root)
    paths = tuple(root.glob(f"*/{WEIGHTS_NAME}"))
    if not paths:
        raise FileNotFoundError(f"no {WEIGHTS_NAME} file was found under {root}.")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def load_generator(
    weights: str | Path,
    *,
    device: torch.device,
) -> Generator:
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
    return build_generator(
        cfg,
        model,
        device=device,
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
        big_vf=geometry.big_vf,
        small_vf=geometry.small_vf,
        big_elongation=geometry.big_elongation,
    )
