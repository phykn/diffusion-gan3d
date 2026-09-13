from collections.abc import Mapping, Sequence

from torch import nn

from src.config import (
    get_domains,
    get_plane_groups,
    get_sr_sizes,
    normalize_train_config,
)
from src.model.critic import ConnectivityCritic2D, PairCritic2D
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.model.sr import SuperResolution


def get_time_scale(cfg: dict) -> float:
    diffusion = cfg["model"]["diffusion"]
    steps = diffusion["num_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("model.diffusion.num_steps must be a positive integer.")
    mode = diffusion["time_embedding"]
    if mode == "index":
        return 1.0
    if mode == "scaled":
        return 1000.0 / steps
    raise ValueError("model.diffusion.time_embedding must be index or scaled.")


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
    cfg = normalize_train_config(cfg)
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    num_domains = len(get_domains(data))
    checkpointing = (
        model["gradient_checkpointing"] if checkpointing is None else checkpointing
    )
    base_channels, multipliers = get_generator_channels(model)
    return Denoiser3D(
        num_phases=data["num_phases"],
        base_channels=base_channels,
        channel_multipliers=multipliers,
        embedding_channels=generator["embedding_channels"],
        latent_channels=generator["latent_channels"],
        num_domains=num_domains,
        gradient_checkpointing=checkpointing,
        anchor_multiscale=generator["anchor_multiscale_input"],
        time_scale=get_time_scale(cfg),
    )


def build_models(
    cfg: dict,
) -> tuple[Denoiser3D, nn.ModuleDict, ConnectivityCritic2D]:
    cfg = normalize_train_config(cfg)
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    critic = model["critic"]
    domains = get_domains(data)
    num_domains = len(domains)
    denoiser = build_denoiser(cfg)
    critics = nn.ModuleDict(
        {
            group: PairCritic2D(
                num_phases=data["num_phases"],
                channels=critic["channels"],
                embedding_channels=generator["embedding_channels"],
                num_domains=num_domains,
                gradient_checkpointing=model["gradient_checkpointing"],
                time_scale=get_time_scale(cfg),
            )
            for group in get_plane_groups(cfg)
        }
    )
    connectivity_critic = ConnectivityCritic2D(
        num_phases=data["num_phases"],
        channels=critic["channels"],
        embedding_channels=generator["embedding_channels"],
        num_domains=num_domains,
        gradient_checkpointing=model["gradient_checkpointing"],
        directed_axis=(
            {"z": 0, "y": 1, "x": 2}.get(data.get("thickness_axis"))
            if cfg["train"]["stability_version"] >= 2
            else None
        ),
    )
    return denoiser, critics, connectivity_critic


def build_diffusion(cfg: dict) -> Diffusion:
    cfg = normalize_train_config(cfg)
    diffusion = cfg["model"]["diffusion"]
    return Diffusion(
        diffusion["num_steps"],
        diffusion["beta_min"],
        diffusion["beta_max"],
    )


def build_sr_model(cfg: dict) -> SuperResolution:
    cfg = normalize_train_config(cfg, "sr")
    get_sr_sizes(cfg)
    return SuperResolution(
        num_phases=cfg["data"]["num_phases"],
        num_domains=len(get_domains(cfg["data"])),
        **cfg["model"]["generator"],
    )
