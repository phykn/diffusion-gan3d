from torch import nn

from src.config.data import get_domains, get_plane_groups, get_sr_plane_groups
from src.config.train import (
    get_sr_resolution,
    get_trainer_sections,
    normalize_train_config,
)
from src.model.critic import PairCritic2D, PlaneCritic2D
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion


def get_time_scale(cfg: dict) -> float:
    diffusion = cfg["model"]["diffusion"]
    if diffusion["time_embedding"] == "index":
        return 1.0
    return 1000.0 / diffusion["num_steps"]


def build_denoiser(cfg: dict) -> Denoiser3D:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    sr = cfg["stage"] == "sr"
    if sr:
        get_sr_resolution(cfg)  # SR models require data.hi_res_size.
    data = cfg["data"]
    model = cfg["model"]
    generator = model["generator"]
    channels = generator["channels"]
    conditioning, _ = get_trainer_sections(cfg)
    return Denoiser3D(
        num_phases=data["num_phases"],
        base_channels=channels[0],
        channel_multipliers=tuple(value // channels[0] for value in channels),
        embedding_channels=generator["embedding_channels"],
        latent_channels=generator["latent_channels"],
        num_domains=len(get_domains(data)),
        gradient_checkpointing=model["gradient_checkpointing"],
        time_scale=get_time_scale(cfg),
        height_enabled=conditioning["height_enabled"],
        coarse_enabled=sr,
        profile_enabled=conditioning["spatial_profile"]["enabled"],
    )


def build_models(cfg: dict) -> tuple[Denoiser3D, nn.ModuleDict]:
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
    conditioning, _ = get_trainer_sections(cfg)
    height = conditioning["height_enabled"]
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
            )
            for group in groups
        }
    )
    return denoiser, critics


def build_diffusion(cfg: dict) -> Diffusion:
    cfg = normalize_train_config(cfg, cfg.get("stage", "low_res"))
    diffusion = cfg["model"]["diffusion"]
    return Diffusion(
        diffusion["num_steps"],
        diffusion["beta_min"],
        diffusion["beta_max"],
    )
