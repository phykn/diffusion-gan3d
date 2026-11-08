from collections.abc import Mapping
from pathlib import Path

import torch

from src.config.schema import StageConfig, build_config, to_dict
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.storage import torch_save

FORMAT = "diffusion-gan3d.export"
VERSION = 2
STAGE_NAMES = {"lr": "LR", "sr": "SR"}
V1_HINT = (
    "v1 runs and exports open only with the v1 code: "
    "git checkout v1 (see README, Changes from v1)."
)


def export_payload(
    config: StageConfig,
    step: int,
    model: Mapping[str, torch.Tensor],
    lr: Mapping | None = None,
) -> dict:
    if (config.stage == "sr") != (lr is not None):
        raise ValueError("an SR export carries its LR export; an LR export does not.")
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "stage": config.stage,
        "step": int(step),
        "config": to_dict(config),
        "model": {
            key: value.detach().to(device="cpu", dtype=torch.float32).clone()
            if value.is_floating_point()
            else value.detach().cpu().clone()
            for key, value in model.items()
        },
    }
    if lr is not None:
        check_export(lr, "lr")
        payload["lr"] = dict(lr)
    return payload


def write_export(path: str | Path, payload: Mapping, *, overwrite=True) -> Path:
    return torch_save(dict(payload), path, overwrite=overwrite)


def resolve_export(weights: str | Path) -> Path:
    path = Path(weights).expanduser()
    if path.is_dir():
        path = path / "generator.pt"
    if not path.is_file():
        raise FileNotFoundError(f"export not found: {path}")
    return path


def read_export(weights: str | Path | Mapping, stage: str) -> dict:
    """Load and check an export: a run folder, a generator.pt or a payload."""
    if isinstance(weights, Mapping):
        return check_export(weights, stage)
    path = resolve_export(weights)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return check_export(payload, stage, path)


def check_export(payload, stage: str, path: Path | str | None = None) -> dict:
    name = STAGE_NAMES[stage]
    where = "" if path is None else f": {path}"
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != FORMAT
        or payload.get("version") != VERSION
        or payload.get("stage") != stage
    ):
        raise ValueError(f"not a v2 {name} export{where}. {V1_HINT}")
    if stage == "sr":
        check_export(payload.get("lr"), "lr", path)
    return dict(payload)


def export_config(payload: Mapping) -> StageConfig:
    return build_config(payload["config"])


def build_denoiser(config: StageConfig) -> Denoiser3D:
    """The stage's denoiser, untrained, from its settings."""
    channels = config.model.generator.channels
    return Denoiser3D(
        num_phases=config.data.num_phases,
        base_channels=channels[0],
        channel_multipliers=tuple(value // channels[0] for value in channels),
        embedding_channels=config.model.generator.embedding_channels,
        latent_channels=config.model.generator.latent_channels,
        gradient_checkpointing=config.model.gradient_checkpointing,
        height_enabled=config.conditioning.height_enabled,
        profile_enabled=config.profile_enabled,
        coarse_enabled=config.stage == "sr",
    )


def build_diffusion(config: StageConfig) -> Diffusion:
    diffusion = config.model.diffusion
    return Diffusion(diffusion.num_steps, diffusion.beta_min, diffusion.beta_max)


def load_denoiser(payload: Mapping, config: StageConfig) -> Denoiser3D:
    """EMA denoiser of an export, in eval mode on the CPU."""
    model = build_denoiser(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)


# Settings an SR model shares with the LR model whose volumes it refines.
SOURCE_KEYS = (
    "data.crop_size",
    "data.lo_res_size",
    "data.hi_res_size",
    "data.num_phases",
    "conditioning.height_enabled",
    "data.height_extent",
)


def check_lr_source(lr: StageConfig, sr: StageConfig) -> None:
    """SR source: the LR settings an SR model depends on must be its own."""
    differences = []
    for key in SOURCE_KEYS:
        section, name = key.split(".")
        mine = getattr(getattr(sr, section), name)
        theirs = getattr(getattr(lr, section), name)
        if mine != theirs:
            differences.append(f"{key} {theirs} (LR) != {mine} (SR)")
    if differences:
        raise ValueError(
            "the LR model does not fit these SR settings: " + "; ".join(differences)
        )
