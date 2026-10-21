"""Resolve presets or saved run settings."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.config.data import (
    DataResolution,
    get_plane_groups,
    get_resolution,
    resolve_split,
)
from src.config.defaults import INACTIVE_DEFAULTS, STAGE_DEFAULTS, TRAIN_DEFAULTS
from src.config.files import PROJECT_ROOT, load_yaml, prepare_yaml
from src.config.schema import validate_config_keys
from src.config.values import _integer, validate_training_values
from src.data.source import fill_depth_profiles, fill_height_extents
from src.plane import PLANE_DIRECTIONS, PLANES

# Settings without defaults; partial configs may omit them until training.
TRAINING_SETTINGS = (
    "data.domains",
    "data.num_phases",
    "data.crop_size",
    "data.lo_res_size",
    "model.generator.channels",
    "model.critic.channels",
    "model.diffusion.num_steps",
    "loss.r1_weight",
    "optim.generator_lr",
    "train.total_steps",
    "train.volume_batch_size",
    "train.real_batch_size",
    "train.slice_pairs_per_plane",
    "train.weights_every_steps",
)


def load_train_config(
    path: str | Path, stage: str = "low_res", data: str | Path | None = None
) -> dict:
    cfg = load_yaml(path)
    if data is not None:
        cfg["data"] = str(data)
    if isinstance(cfg.get("data"), str):
        cfg["data"] = load_yaml(resolve_external_data_path(cfg["data"]))
    cfg = fill_height_extents(normalize_loaded_train_config(cfg, stage))
    return fill_depth_profiles(cfg)


def resolve_external_data_path(path: str | Path) -> Path:
    data_path = Path(path).expanduser()
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return data_path


def normalize_loaded_train_config(cfg: dict, stage: str = "low_res") -> dict:
    if "data" not in cfg:
        raise ValueError(
            "training config must select a data file or embed data settings."
        )
    if stage == "sr" and "guidance" not in cfg.get("lr_bank", {}):
        raise ValueError("SR config must set lr_bank.guidance explicitly.")
    cfg = normalize_train_config(cfg, stage)
    # Relative folders follow the project root, like every other config path.
    cfg["data"]["domains"] = {
        domain: {
            plane: [
                str((PROJECT_ROOT / Path(path).expanduser()).resolve())
                for path in paths
            ]
            for plane, paths in planes.items()
        }
        for domain, planes in cfg["data"]["domains"].items()
    }
    return cfg


def normalize_train_config(cfg: Mapping, stage: str = "low_res") -> dict:
    if stage not in ("low_res", "sr") or cfg.get("stage", stage) != stage:
        raise ValueError(f"expected stage: {stage}.")
    cfg = validate_config_keys(cfg, stage)
    cfg = prepare_yaml(cfg)
    cfg["stage"] = stage
    if "split" in cfg.get("data", {}):
        cfg["data"]["split"] = resolve_split(cfg["data"]["split"], PROJECT_ROOT)
    _fill_defaults(cfg, TRAIN_DEFAULTS | STAGE_DEFAULTS[stage])
    nickname = cfg["nickname"]
    if nickname is None:
        nickname = ""
    if not isinstance(nickname, str):
        raise ValueError("nickname must be a string or null.")
    nickname = nickname.strip()
    if nickname.endswith(".") or any(
        char in '<>:"/\\|?*' or ord(char) < 32 for char in nickname
    ):
        raise ValueError("nickname must be a valid folder-name component.")
    cfg["nickname"] = nickname
    flags = [
        ("conditioning.height_enabled", cfg["conditioning"]["height_enabled"]),
        ("train.mixed_precision", cfg["train"]["mixed_precision"]),
        ("model.gradient_checkpointing", cfg["model"]["gradient_checkpointing"]),
        ("augmentation.real_only", cfg["augmentation"]["real_only"]),
    ]
    if stage == "low_res":
        name = "model.generator.anchor_multiscale_input"
        flags.append((name, cfg["model"]["generator"]["anchor_multiscale_input"]))
    for name, value in flags:
        # YAML reads a quoted "false" as text, which would act as true.
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    for name in ("embedding_channels", "latent_channels"):
        _integer(cfg["model"]["generator"][name], f"model.generator.{name}")
    channels = cfg["model"]["critic"].get("channels")
    if channels is not None and (
        not isinstance(channels, list)
        or not channels
        or any(type(value) is not int or value < 1 for value in channels)
    ):
        raise ValueError("model.critic.channels must list positive integers.")
    augmentation = cfg["augmentation"]
    auto_planes = augmentation.get("auto_planes", False)
    if type(auto_planes) is not bool:
        raise ValueError("augmentation.auto_planes must be a boolean.")
    if auto_planes:
        if "planes" in augmentation:
            raise ValueError("choose augmentation.auto_planes or explicit planes.")
        height_axis = "z" if cfg["conditioning"]["height_enabled"] else None
        augmentation["planes"] = {
            plane: {
                "flip_axes": [axis for axis in axes if axis != height_axis],
                "rotate_90": height_axis not in axes,
            }
            for plane, axes in PLANE_DIRECTIONS.items()
        }
        # Store resolved policies so checkpoint behavior is independent of defaults.
        augmentation.pop("auto_planes")
    _integer(cfg["train"]["structure_every_steps"], "train.structure_every_steps", 0)
    _integer(
        cfg["model"]["critic"]["pyramid_min_size"], "model.critic.pyramid_min_size", 2
    )
    if cfg["model"]["critic"]["input_mode"] not in ("pair", "single"):
        raise ValueError("model.critic.input_mode must be pair or single.")
    statistics = cfg["loss"]["group_statistics"]
    _integer(statistics["max_gap"], "loss.group_statistics.max_gap")
    get_schedule_steps(statistics, "loss.group_statistics")
    if stage == "low_res":
        loss = cfg["loss"]
        _integer(loss["anchor_neighborhood_radius"], "loss.anchor_neighborhood_radius")
        _integer(loss["anchor_seam_rows"], "loss.anchor_seam_rows")
        _integer(
            loss["connectivity"]["max_slice_gap"], "loss.connectivity.max_slice_gap"
        )
        profile = cfg["conditioning"]["spatial_profile"]
        if type(profile["enabled"]) is not bool:
            raise ValueError("spatial_profile.enabled must be a boolean.")
        if type(profile["critic_enabled"]) is not bool or (
            profile["critic_enabled"] and not profile["enabled"]
        ):
            raise ValueError(
                "spatial_profile.critic_enabled requires an enabled profile."
            )
        _integer(profile["num_bins"], "spatial_profile.num_bins")
        if profile["enabled"] and not cfg["conditioning"]["height_enabled"]:
            raise ValueError("spatial_profile requires height_enabled.")
    optim = cfg["optim"]
    if "critic_lr" not in optim and "generator_lr" in optim:
        optim["critic_lr"] = optim["generator_lr"]
    validate_training_values(cfg)
    return cfg


def validate_training_settings(cfg: Mapping) -> None:
    """Check settings used only by training, before data or models are built."""
    _require_settings(cfg, TRAINING_SETTINGS, " for training")
    train = cfg["train"]
    conditioning, _ = get_trainer_sections(cfg)
    anchor = conditioning["anchor"]
    start, _ = get_schedule_steps(anchor, "conditioning.anchor")
    if train["volume_batch_size"] > train["real_batch_size"]:
        if anchor["probability"] > 0.0 and start < train["total_steps"]:
            raise ValueError(
                "train.volume_batch_size must not exceed train.real_batch_size when "
                "anchor training is enabled."
            )
        if conditioning["height_enabled"]:
            raise ValueError(
                "height conditioning requires real_batch_size >= volume_batch_size."
            )


def _require_settings(cfg: Mapping, paths: Sequence[str], purpose: str = "") -> None:
    for path in paths:
        target = cfg
        for key in path.split("."):
            if not isinstance(target, Mapping) or key not in target:
                raise ValueError(f"missing required setting{purpose}: {path}")
            target = target[key]


def _fill_defaults(cfg: dict, defaults: Mapping) -> None:
    for path, value in defaults.items():
        target = cfg
        *sections, key = path.split(".")
        for section in sections:
            target = target.setdefault(section, {})
        if key not in target:
            target[key] = prepare_yaml(value)


def get_trainer_sections(cfg: Mapping) -> tuple[dict, dict]:
    """Complete normalized conditioning/loss for the shared LR/SR trainer.

    Inactive options are runtime-only; do not broaden the stage's YAML schema
    or mutate the resolved snapshot used for resume validation.
    """
    sections = prepare_yaml({"conditioning": cfg["conditioning"], "loss": cfg["loss"]})
    _fill_defaults(sections, INACTIVE_DEFAULTS[cfg["stage"]])
    return sections["conditioning"], sections["loss"]


def validate_sr_config(cfg: dict) -> dict:
    cfg = normalize_train_config(cfg, "sr")
    data_sha256 = cfg.get("source", {}).get("data_sha256")
    if data_sha256 is not None and (
        not isinstance(data_sha256, str)
        or len(data_sha256) != 64
        or any(char not in "0123456789abcdef" for char in data_sha256)
    ):
        raise ValueError("source.data_sha256 must be a SHA-256 hex digest.")
    get_sr_resolution(cfg)  # Requires data.hi_res_size.
    groups = get_plane_groups(cfg)
    cfg["model"]["critic"]["plane_groups"] = [
        [PLANES[axis] for axis in axes] for axes in groups.values()
    ]
    validate_training_settings(cfg)
    _require_settings(cfg, ("lr_bank.samples_per_domain",), " for training")
    _integer(cfg["lr_bank"]["samples_per_domain"], "lr_bank.samples_per_domain")
    _integer(cfg["lr_bank"]["refresh_every_steps"], "lr_bank.refresh_every_steps", 0)
    guidance = cfg["lr_bank"]["guidance"]
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
    ):
        raise ValueError("lr_bank.guidance must be finite.")
    return cfg


def get_sr_resolution(cfg: Mapping) -> DataResolution:
    if "hi_res_size" not in cfg["data"]:
        raise ValueError("SR requires data.hi_res_size.")
    return get_resolution(cfg["data"])


def get_schedule_steps(
    section: dict,
    name: str,
) -> tuple[int, int]:
    start = section["start_step"]
    ramp = section["ramp_steps"]
    for field, value in (("start_step", start), ("ramp_steps", ramp)):
        _integer(value, f"{name}.{field}", 0)
    return start, ramp
