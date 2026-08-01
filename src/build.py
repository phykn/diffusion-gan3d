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
from .train.ema import build_ema
from .train.engine import Trainer
from .train.weights import load_weights
from .utils import load_yaml

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def build_trainer(cfg: dict, device: torch.device) -> Trainer:
    denoiser, critics = build_models(cfg)
    load_checkpoints(cfg, denoiser, critics)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(denoiser, critics, cfg)
    data = cfg["data"]
    model = cfg["model"]
    anchor = cfg["anchor"]
    vf = cfg["vf"]
    optim = cfg["optim"]
    train = cfg["train"]
    use_amp = train["mixed_precision"] and device.type == "cuda"
    return Trainer(
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=build_streams(cfg, device),
        diffusion=build_diffusion(cfg).to(device),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
        device=device,
        volume_batch_size=train["volume_batch_size"],
        volume_sizes=train["volume_sizes"],
        num_phases=data["num_phases"],
        patch_size=data["patch_size"],
        slices_per_axis=train["slices_per_axis"],
        ema_decay=train["ema_decay"],
        r1_gamma=optim["r1_gamma"],
        r1_interval=optim["r1_interval"],
        critic_local_weight=optim.get("critic_local_weight", 0.5),
        anchor_dropout=anchor["dropout"],
        anchor_loss_weight=anchor["loss_weight"],
        anchor_max_planes=anchor.get("max_planes", 1),
        vf_loss_weight=vf["loss_weight"],
        vf_dropout=vf["dropout"],
        latent_channels=model["latent_channels"],
        amp_enabled=use_amp,
    )


def build_models(cfg: dict) -> tuple[Denoiser3D, nn.ModuleDict]:
    data = cfg["data"]
    model = cfg["model"]
    denoiser = build_denoiser(cfg)
    critics = nn.ModuleDict(
        {
            str(axis): PairCritic2D(
                num_phases=data["num_phases"],
                channels=model["critic_channels"],
                embedding_channels=model["embedding_channels"],
                gradient_checkpointing=model.get("gradient_checkpointing", True),
            )
            for axis in AXES
        }
    )
    return denoiser, critics


def build_denoiser(
    cfg: dict,
    checkpointing: bool | None = None,
) -> Denoiser3D:
    data = cfg["data"]
    model = cfg["model"]
    checkpointing = (
        model.get("gradient_checkpointing", True)
        if checkpointing is None
        else checkpointing
    )
    return Denoiser3D(
        num_phases=data["num_phases"],
        base_channels=model["base_channels"],
        channel_multipliers=model["channel_multipliers"],
        embedding_channels=model["embedding_channels"],
        latent_channels=model["latent_channels"],
        gradient_checkpointing=checkpointing,
    )


def build_optimizers(
    denoiser: nn.Module,
    critics: nn.ModuleDict,
    cfg: dict,
) -> tuple[torch.optim.Optimizer, dict[str, torch.optim.Optimizer]]:
    optim = cfg["optim"]
    betas = (optim["beta1"], optim["beta2"])
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
        for axis in AXES
    }
    return denoiser_optim, critic_optims


def load_checkpoints(
    cfg: dict,
    denoiser: nn.Module,
    critics: nn.ModuleDict,
) -> None:
    checkpoints = cfg["train"].get("checkpoint") or {}
    if checkpoints.get("model") is not None:
        load_weights(checkpoints["model"], denoiser)
    for axis in AXES:
        path = checkpoints.get(f"critic_{axis}")
        if path is not None:
            load_weights(path, critics[str(axis)])


def build_streams(
    cfg: dict,
    device: torch.device,
) -> dict[int, BatchStream]:
    datasets = build_datasets(cfg)
    data = cfg["data"]
    return {
        axis: build_stream(
            datasets[axis],
            data["batch_size"],
            data.get("num_workers", 0),
            device.type == "cuda",
        )
        for axis in AXES
    }


def build_datasets(cfg: dict) -> dict[int, SliceDataset]:
    data = cfg["data"]
    grouped = find_slices(data["folder"])
    return {
        axis: SliceDataset(
            grouped[axis],
            crop_size=data["crop_size"],
            patch_size=data["patch_size"],
        )
        for axis in AXES
    }


def find_slices(
    folders: dict[int, str | Path],
) -> dict[int, tuple[Path, ...]]:
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
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )
        if not paths:
            raise ValueError(f"axis {axis} folder contains no images: {folder}")
        grouped[axis] = paths
    return grouped


def build_stream(
    dataset: SliceDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=max(len(dataset), batch_size),
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


def build_diffusion(cfg: dict) -> Diffusion:
    diffusion = cfg["diffusion"]
    return Diffusion(
        diffusion["timesteps"],
        diffusion["beta_min"],
        diffusion["beta_max"],
    )


def load_generator(
    weights: str | Path,
    device: torch.device,
) -> Generator:
    path = Path(weights).resolve()
    cfg = load_yaml(path.parent / "train.yaml")
    denoiser = build_denoiser(cfg, checkpointing=False).to(device)
    try:
        load_weights(path, denoiser)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    denoiser.eval()
    return build_generator(cfg, denoiser, device)


def build_generator(
    cfg: dict,
    denoiser: Denoiser3D,
    device: torch.device,
) -> Generator:
    data = cfg["data"]
    model = cfg["model"]
    anchor = cfg["anchor"]
    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    return Generator(
        denoiser,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=data["patch_size"],
        num_phases=data["num_phases"],
        latent_channels=model["latent_channels"],
        anchor_enabled=anchor["dropout"] < 1.0,
        use_amp=use_amp,
    )
