from collections.abc import Mapping, Sequence

from torch import nn

from src.config.data import get_domains, get_plane_groups, get_sr_plane_groups
from src.config.train import get_sr_resolution, normalize_train_config
from src.model.critic import (
    ConnectivityCritic2D,
    PairCritic2D,
    PlaneCritic2D,
    SeamCritic2D,
)
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion


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
        not isinstance(value, int) or isinstance(value, bool) or value < 2
        for value in values
    ):
        raise ValueError(
            "model.generator.channels must contain integers of at least two."
        )
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
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
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
        anchor_multiscale=generator.get("anchor_multiscale_input", False),
        time_scale=get_time_scale(cfg),
        height_enabled=cfg["conditioning"]["height_enabled"],
        coarse_enabled=cfg["stage"] == "sr",
        profile_enabled=cfg["conditioning"]
        .get("spatial_profile", {})
        .get("enabled", False),
    )


def build_models(
    cfg: dict,
) -> tuple[Denoiser3D, nn.ModuleDict, ConnectivityCritic2D | None]:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    critic = model["critic"]
    domains = get_domains(data)
    num_domains = len(domains)
    denoiser = build_denoiser(cfg)
    if cfg["stage"] == "sr":
        groups = {
            group: axes
            for domain_groups in get_sr_plane_groups(cfg).values()
            for group, axes in domain_groups.items()
        }
    else:
        groups = get_plane_groups(cfg)
    height = cfg["conditioning"]["height_enabled"]
    profile = (
        cfg["conditioning"].get("spatial_profile", {}).get("critic_enabled", False)
    )
    phases = data["num_phases"]
    shared = dict(
        num_phases=phases,
        channels=critic["channels"],
        embedding_channels=generator["embedding_channels"],
        num_domains=num_domains,
        gradient_checkpointing=model["gradient_checkpointing"],
        pyramid_min_size=critic["pyramid_min_size"],
    )
    critic_class = PlaneCritic2D if critic["input_mode"] == "single" else PairCritic2D
    critics = nn.ModuleDict(
        {
            group: critic_class(
                **shared,
                time_scale=get_time_scale(cfg),
                height_channels=int(height),
                profile_channels=phases if profile else 0,
            )
            for group in groups
        }
    )
    # Connectivity triplets stack three slices, so their conditions do too.
    connectivity_critic = (
        None
        if cfg["stage"] == "sr"
        else ConnectivityCritic2D(
            **shared,
            directed_axis=0 if height else None,
            height_channels=3 if height else 0,
            profile_channels=3 * phases if profile else 0,
        )
    )
    return denoiser, critics, connectivity_critic


def build_seam_critic(cfg: dict) -> SeamCritic2D | None:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    if cfg["stage"] == "sr":
        return None
    critic = cfg["model"]["critic"]
    return SeamCritic2D(
        num_phases=cfg["data"]["num_phases"],
        channels=critic["channels"],
        embedding_channels=cfg["model"]["generator"]["embedding_channels"],
        num_domains=len(get_domains(cfg["data"])),
        gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
        time_scale=get_time_scale(cfg),
        pyramid_min_size=critic["pyramid_min_size"],
    )


def build_diffusion(cfg: dict) -> Diffusion:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    diffusion = cfg["model"]["diffusion"]
    return Diffusion(
        diffusion["num_steps"],
        diffusion["beta_min"],
        diffusion["beta_max"],
    )


def build_sr_model(cfg: dict) -> Denoiser3D:
    cfg = normalize_train_config(cfg, "sr")
    get_sr_resolution(cfg)  # SR models require data.hi_res_size.
    return build_denoiser(cfg, checkpointing=False)
