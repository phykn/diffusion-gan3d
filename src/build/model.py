from collections.abc import Mapping, Sequence

from torch import nn

from src.config.data import get_domains, get_plane_groups
from src.config.train import get_sr_sizes, normalize_train_config
from src.model.critic import ConnectivityCritic2D, PairCritic2D
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
    groups = get_plane_groups(cfg)
    if cfg["stage"] == "sr":
        groups = {
            f"{domain}_{group}": axes
            for domain, planes in domains.items()
            for group, axes in groups.items()
            if set(axes).intersection(planes)
        }
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
            for group in groups
        }
    )
    connectivity_critic = (
        None
        if cfg["stage"] == "sr"
        else ConnectivityCritic2D(
            num_phases=data["num_phases"],
            channels=critic["channels"],
            embedding_channels=generator["embedding_channels"],
            num_domains=num_domains,
            gradient_checkpointing=model["gradient_checkpointing"],
            directed_axis={"z": 0, "y": 1, "x": 2}.get(data.get("thickness_axis")),
        )
    )
    if cfg["conditioning"]["height_enabled"]:
        for network in critics.values():
            network.height_input = nn.Conv2d(
                1, critic["channels"][0], 3, padding=1, bias=False
            )
    if cfg["conditioning"]["height_enabled"] and connectivity_critic is not None:
        connectivity_critic.height_input = nn.Conv2d(
            3, critic["channels"][0], 3, padding=1, bias=False
        )
    if cfg["conditioning"].get("spatial_profile", {}).get("critic_enabled", False):
        for network in critics.values():
            network.profile_input = nn.Conv2d(
                data["num_phases"], critic["channels"][0], 3, padding=1, bias=False
            )
        connectivity_critic.profile_input = nn.Conv2d(
            3 * data["num_phases"], critic["channels"][0], 3, padding=1, bias=False
        )
    for network in (*critics.values(), connectivity_critic):
        if network is not None:
            network.pyramid_min_size = critic["pyramid_min_size"]
    return denoiser, critics, connectivity_critic


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
    data = cfg["data"]
    data["crop_size"], data["lo_res_size"], data["hi_res_size"] = get_sr_sizes(cfg)
    return build_denoiser(cfg, checkpointing=False)
