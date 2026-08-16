import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import AXES
from .config import find_train_config, get_schedule_steps
from .dataset import BatchStream, FolderBatchSampler, SliceDataset
from .diffusion import Diffusion
from .generate import Generator
from .model.critic import ConnectivityCritic2D, PairCritic2D
from .model.denoiser import Denoiser3D
from .train.augment import CriticAugment
from .train.ema import build_ema
from .train.engine import Trainer, TrainerComponents, TrainerSettings
from .train.weights import load_all_weights, load_weights
from .utils import load_yaml


def get_domains(
    data: Mapping[str, object],
) -> dict[int, dict[int, Sequence[str | Path]]]:
    domains = data["domains"]
    if not isinstance(domains, Mapping):
        raise TypeError("data.domains must be a mapping.")
    if not domains:
        raise ValueError("data.domains must not be empty.")
    if set(domains) != set(range(len(domains))):
        raise ValueError("domain IDs must be contiguous and start at zero.")
    parsed = {}
    for domain, folders in domains.items():
        if not isinstance(folders, Mapping):
            raise TypeError(f"domain {domain} must map axes to folders.")
        if not folders:
            raise ValueError(f"domain {domain} must contain at least one axis.")
        unknown = set(folders) - set(AXES)
        if unknown:
            raise ValueError(
                f"domain {domain} contains invalid axes: {sorted(unknown)}."
            )
        parsed[domain] = dict(folders)
    return parsed


def get_data_axes(data: Mapping[str, object]) -> tuple[int, ...]:
    domains = get_domains(data)
    return tuple(
        axis for axis in AXES if any(axis in folders for folders in domains.values())
    )


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


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def find_slice_groups(
    folders: dict[int, Sequence[str | Path]],
) -> dict[int, tuple[tuple[Path, ...], ...]]:
    if not folders:
        raise ValueError("axis folders must contain at least one axis.")
    if not set(folders).issubset(AXES):
        raise ValueError("axis folders may contain only axes 0, 1, and 2.")

    grouped = {}
    for axis in sorted(folders):
        values = folders[axis]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"axis {axis} folders must be a sequence of paths.")
        if not values:
            raise ValueError(f"axis {axis} folders must not be empty.")
        axis_folders = tuple(Path(value) for value in values)
        if len({folder.resolve() for folder in axis_folders}) != len(axis_folders):
            raise ValueError(f"axis {axis} folders must not contain duplicates.")

        groups = []
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
            groups.append(tuple(found))
        grouped[axis] = tuple(groups)
    return grouped


def find_slices(
    folders: dict[int, Sequence[str | Path]],
) -> dict[int, tuple[Path, ...]]:
    groups = find_slice_groups(folders)
    return {
        axis: tuple(path for group in path_groups for path in group)
        for axis, path_groups in groups.items()
    }


def build_datasets(cfg: dict) -> dict[int, dict[int, SliceDataset]]:
    data = cfg["data"]
    return {
        domain_id: {
            axis: SliceDataset.from_path_groups(
                path_groups,
                crop_size=data["crop_size"],
                patch_size=data["input_size"],
                allow_partial_crop=data["crop_partial"],
            )
            for axis, path_groups in find_slice_groups(folders).items()
        }
        for domain_id, folders in get_domains(data).items()
    }


def build_stream(
    dataset: SliceDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    loader = DataLoader(
        dataset,
        batch_sampler=FolderBatchSampler(dataset, batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return BatchStream(loader)


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
    num_domains = len(get_domains(data))
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
            for axis in get_data_axes(data)
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
        load_weights(path, denoiser)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    denoiser.eval()
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    use_amp = train["amp"] and device.type == "cuda"
    return Generator(
        denoiser,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=data["input_size"],
        num_phases=data["num_phase"],
        latent_channels=model["generator"]["latent_channels"],
        use_amp=use_amp,
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
    validate_prior_capacity(
        data=data,
        train=train,
        anchor=anchor,
        connectivity=connectivity,
        anchor_start_step=anchor_start_step,
        anchor_ramp_steps=anchor_ramp_steps,
    )
    denoiser, critics, connectivity_critic = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    connectivity_critic = connectivity_critic.to(device)
    ema = build_ema(denoiser)
    initial_weights = train.get("init_weights")
    if initial_weights is not None:
        load_all_weights(
            initial_weights,
            denoiser,
            ema,
            critics,
            connectivity_critic,
        )
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
            connectivity_bank_size=connectivity["volume_count"],
            connectivity_refresh_steps=connectivity["refresh_every"],
            vf_loss_weight=vf["weight"],
            vf_target_average_max_samples=vf["max_samples"],
            domain_dropout=1.0 - data["domain_prob"],
            cfg_drop_each_probability=conditioning["joint_each_prob"],
            latent_channels=generator["latent_channels"],
            amp_enabled=use_amp,
        ),
    )


def validate_prior_capacity(
    *,
    data: dict,
    train: dict,
    anchor: dict,
    connectivity: dict,
    anchor_start_step: int,
    anchor_ramp_steps: int,
) -> None:
    anchor_active = anchor["train_prob"] > 0.0 and anchor_start_step < train["steps"]
    if not anchor_active:
        return
    if train["volume_batch_size"] > data["batch_size"]:
        raise ValueError(
            "train.volume_batch_size must not exceed data.batch_size when "
            "anchor training is enabled."
        )
    if connectivity["weight"] <= 0.0 and connectivity["phase_transition_weight"] <= 0.0:
        return

    bank_size = connectivity["volume_count"]
    if not isinstance(bank_size, int) or isinstance(bank_size, bool) or bank_size < 1:
        raise ValueError("anchor.connectivity.volume_count must be a positive integer.")
    volume_batch_size = train["volume_batch_size"]
    if (
        not isinstance(volume_batch_size, int)
        or isinstance(volume_batch_size, bool)
        or volume_batch_size < 1
    ):
        raise ValueError("train.volume_batch_size must be a positive integer.")
    build_steps = len(get_domains(data)) * math.ceil(bank_size / volume_batch_size)
    available_steps = train["steps"] - anchor_start_step
    # The conditional prior is collected only after the real-anchor ramp has
    # completed. One following step is then needed before a ready bank can be
    # sampled as a multi-plane condition.
    required_steps = max(anchor_ramp_steps, 1) + build_steps
    if available_steps < required_steps:
        raise ValueError(
            "training must leave enough steps after anchor.start_step to fill "
            "every conditional prior bank and leave one multi-anchor-eligible step; "
            f"need {required_steps}, got {available_steps}."
        )
