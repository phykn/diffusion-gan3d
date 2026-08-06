from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from . import AXES
from .dataset import BatchStream, SliceDataset
from .diffusion import Diffusion
from .generate import Generator
from .model.critic import ConnectivityCritic2D, PairCritic2D
from .model.denoiser import Denoiser3D
from .train.augment import CriticAugment
from .train.ema import build_ema
from .train.engine import Trainer
from .train.weights import load_weights
from .utils import load_yaml

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def get_config_value(section: dict, name: str, legacy_name: str) -> object:
    if name in section and legacy_name in section:
        raise ValueError(f"use only {name!r}; remove legacy {legacy_name!r}.")
    if name in section:
        return section[name]
    if legacy_name in section:
        return section[legacy_name]
    raise KeyError(name)


def get_anchor_steps(anchor: dict, total_steps: int) -> tuple[int, int]:
    uses_steps = "start_step" in anchor or "ramp_steps" in anchor
    uses_ratios = "start_ratio" in anchor or "ramp_ratio" in anchor
    if uses_steps and uses_ratios:
        raise ValueError("anchor step and ratio fields cannot be mixed.")
    if uses_steps:
        start = anchor.get("start_step")
        ramp = anchor.get("ramp_steps")
        for name, value in (("start_step", start), ("ramp_steps", ramp)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"anchor.{name} must be a non-negative integer.")
        if start + ramp > total_steps:
            raise ValueError("anchor start and ramp steps must not exceed total steps.")
        return start, ramp

    start_ratio = anchor.get("start_ratio")
    ramp_ratio = anchor.get("ramp_ratio")
    for name, value in (("start_ratio", start_ratio), ("ramp_ratio", ramp_ratio)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"anchor.{name} must be between zero and one.")
    if start_ratio + ramp_ratio > 1.0:
        raise ValueError("anchor start and ramp ratios must not exceed one.")
    return round(total_steps * start_ratio), round(total_steps * ramp_ratio)


def find_slices(
    folders: dict[int, Sequence[str | Path]],
) -> dict[int, tuple[Path, ...]]:
    if set(folders) != set(AXES):
        raise ValueError("axis folders must contain exactly axes 0, 1, and 2.")

    grouped = {}
    for axis in AXES:
        values = folders[axis]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"axis {axis} folders must be a sequence of paths.")
        if not values:
            raise ValueError(f"axis {axis} folders must not be empty.")
        axis_folders = tuple(Path(value) for value in values)
        if len({folder.resolve() for folder in axis_folders}) != len(axis_folders):
            raise ValueError(f"axis {axis} folders must not contain duplicates.")

        paths = []
        for folder in axis_folders:
            if not folder.is_dir():
                raise FileNotFoundError(f"axis {axis} folder does not exist: {folder}")
            found = sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not found:
                raise ValueError(f"axis {axis} folder contains no images: {folder}")
            paths.extend(found)
        grouped[axis] = tuple(paths)
    return grouped


def build_datasets(cfg: dict) -> dict[int, SliceDataset]:
    data = cfg["data"]
    grouped = find_slices(data["folders"])
    return {
        axis: SliceDataset(
            grouped[axis],
            crop_size=data["crop_size"],
            patch_size=data["input_size"],
        )
        for axis in AXES
    }


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


def build_denoiser(
    cfg: dict,
    checkpointing: bool | None = None,
) -> Denoiser3D:
    data = cfg["data"]
    model = cfg["model"]
    checkpointing = (
        model["gradient_checkpointing"] if checkpointing is None else checkpointing
    )
    return Denoiser3D(
        num_phases=data["num_phases"],
        base_channels=model["base_channels"],
        channel_multipliers=model["channel_multipliers"],
        embedding_channels=model["embedding_channels"],
        latent_channels=model["latent_channels"],
        gradient_checkpointing=checkpointing,
    )


def build_models(
    cfg: dict,
) -> tuple[Denoiser3D, nn.ModuleDict, ConnectivityCritic2D]:
    data = cfg["data"]
    model = cfg["model"]
    connectivity = cfg["connectivity"]
    denoiser = build_denoiser(cfg)
    critics = nn.ModuleDict(
        {
            str(axis): PairCritic2D(
                num_phases=data["num_phases"],
                channels=model["critic_channels"],
                embedding_channels=model["embedding_channels"],
                gradient_checkpointing=model["gradient_checkpointing"],
            )
            for axis in AXES
        }
    )
    connectivity_critic = ConnectivityCritic2D(
        num_phases=data["num_phases"],
        channels=model["critic_channels"],
        embedding_channels=model["embedding_channels"],
        reversal_invariant=connectivity["reversal_invariant"],
        gradient_checkpointing=model["gradient_checkpointing"],
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
    betas = (optim["beta1"], optim["beta2"])
    denoiser_optim = torch.optim.Adam(
        denoiser.parameters(),
        lr=optim["denoiser_lr"],
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
    connectivity_optim = torch.optim.Adam(
        connectivity_critic.parameters(),
        lr=optim["critic_lr"],
        betas=betas,
    )
    return denoiser_optim, critic_optims, connectivity_optim


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
    config = next(
        (
            parent / "train.yaml"
            for parent in path.parents
            if (parent / "train.yaml").is_file()
        ),
        None,
    )
    if config is None:
        raise FileNotFoundError(f"train.yaml was not found above weights file: {path}")
    cfg = load_yaml(config)
    denoiser = build_denoiser(cfg, checkpointing=False).to(device)
    try:
        load_weights(path, denoiser)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    denoiser.eval()
    data = cfg["data"]
    model = cfg["model"]
    anchor = cfg["anchor"]
    train = cfg["train"]
    use_amp = train["mixed_precision"] and device.type == "cuda"
    anchor_start_step, _ = get_anchor_steps(anchor, train["total_steps"])
    anchor_dropout = get_config_value(
        anchor,
        "dropout_prob",
        "dropout_probability",
    )
    return Generator(
        denoiser,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=data["input_size"],
        num_phases=data["num_phases"],
        latent_channels=model["latent_channels"],
        anchor_enabled=(
            anchor_dropout < 1.0 and anchor_start_step < train["total_steps"]
        ),
        use_amp=use_amp,
    )


def build_trainer(cfg: dict, device: torch.device) -> Trainer:
    denoiser, critics, connectivity_critic = build_models(cfg)
    train = cfg["train"]
    if train.get("generator") is not None:
        load_weights(train["generator"], denoiser)
    for axis in AXES:
        path = train.get(f"critic_{axis}")
        if path is not None:
            load_weights(path, critics[str(axis)])
    if train.get("critic_c") is not None:
        load_weights(train["critic_c"], connectivity_critic)

    denoiser = denoiser.to(device)
    critics = critics.to(device)
    connectivity_critic = connectivity_critic.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    data = cfg["data"]
    model = cfg["model"]
    anchor = cfg["anchor"]
    connectivity = cfg["connectivity"]
    vf = cfg["vf"]
    optim = cfg["optim"]
    anchor_start_step, anchor_ramp_steps = get_anchor_steps(
        anchor,
        train["total_steps"],
    )
    anchor_dropout = get_config_value(
        anchor,
        "dropout_prob",
        "dropout_probability",
    )
    anchor_multi_prob = get_config_value(
        anchor,
        "multi_anchor_prob",
        "multi_anchor_probability",
    )
    anchor_mixed_axis_prob = get_config_value(
        anchor,
        "mixed_axis_prob",
        "mixed_axis_probability",
    )
    vf_dropout = get_config_value(vf, "dropout_prob", "dropout_probability")
    critic_augment = CriticAugment(
        data.get("augment", False),
        prob=data.get("augment_prob", 1.0),
    )
    use_amp = train["mixed_precision"] and device.type == "cuda"
    datasets = build_datasets(cfg)
    streams = {
        axis: build_stream(
            datasets[axis],
            batch_size=data["batch_size"],
            num_workers=data["num_workers"],
            pin_memory=device.type == "cuda",
        )
        for axis in AXES
    }

    return Trainer(
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
        volume_batch_size=train["volume_batch_size"],
        volume_sizes=train["volume_sizes"],
        num_phases=data["num_phases"],
        patch_size=data["input_size"],
        slice_pairs_per_axis=train["slice_pairs_per_axis"],
        ema_decay=train["ema_decay"],
        r1_gamma=optim["r1_gamma"],
        r1_interval=optim["r1_interval"],
        critic_local_weight=optim["local_loss_weight"],
        anchor_dropout=anchor_dropout,
        anchor_start_step=anchor_start_step,
        anchor_ramp_steps=anchor_ramp_steps,
        anchor_multi_probability=anchor_multi_prob,
        anchor_max_density=anchor["max_density"],
        anchor_min_spacing=anchor["min_spacing"],
        anchor_mixed_axis_probability=anchor_mixed_axis_prob,
        anchor_teacher_bank_mebibytes=anchor["teacher_bank_size_mib"],
        anchor_loss_weight=anchor["loss_weight"],
        connectivity_weight=connectivity["loss_weight"],
        connectivity_replay_triplets_per_axis=connectivity["replay_triplets_per_axis"],
        connectivity_replay_capacity_per_axis=connectivity["replay_capacity_per_axis"],
        connectivity_max_triplets_per_step=connectivity["max_triplets_per_step"],
        vf_loss_weight=vf["loss_weight"],
        vf_dropout=vf_dropout,
        latent_channels=model["latent_channels"],
        amp_enabled=use_amp,
        critic_augment=critic_augment,
    )
