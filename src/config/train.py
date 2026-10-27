import copy
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
from src.config.files import PROJECT_ROOT, load_yaml, prepare_yaml, project_path
from src.config.schema import validate_config_keys
from src.config.values import (
    check_boolean,
    check_integer,
    validate_training_values,
)
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


def load_train_config(config: str | Path | Mapping, stage: str = "low_res") -> dict:
    """Resolve a training YAML path, or a config built by src.config.sample."""
    if isinstance(config, Mapping):
        cfg = copy.deepcopy(dict(config))
    else:
        cfg = load_yaml(config)
    cfg = fill_height_extents(normalize_loaded_train_config(cfg, stage))
    return fill_depth_profiles(cfg)


def normalize_loaded_train_config(cfg: dict, stage: str = "low_res") -> dict:
    if "data" not in cfg:
        raise ValueError("training config must embed its data settings.")
    if stage == "sr" and "guidance" not in cfg.get("lr_bank", {}):
        raise ValueError("SR config must set lr_bank.guidance explicitly.")
    cfg = normalize_train_config(cfg, stage)
    # Relative folders follow the project root, like every other config path.
    cfg["data"]["domains"] = {
        domain: {
            plane: [str(project_path(path)) for path in paths]
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
        "conditioning.height_enabled",
        "train.mixed_precision",
        "model.gradient_checkpointing",
        "augmentation.real_only",
        "selection.enabled",
    ]
    if stage == "low_res":
        flags += [
            "conditioning.spatial_profile.enabled",
            "conditioning.spatial_profile.critic_enabled",
        ]
    for name in flags:
        check_boolean(_setting(cfg, name), name)
    integers = [
        ("model.generator.embedding_channels", 1),
        ("model.generator.latent_channels", 1),
        ("train.structure_every_steps", 0),
        ("model.critic.pyramid_min_size", 2),
        ("loss.group_statistics.max_gap", 1),
        ("loss.mirror_balance.max_gap", 1),
        ("selection.samples", 1),
    ]
    if stage == "low_res":
        integers += [
            ("loss.real_transition.max_gap", 1),
            ("conditioning.spatial_profile.num_bins", 1),
        ]
    for name, minimum in integers:
        check_integer(_setting(cfg, name), name, minimum)
    # Generator widths are normalized by channel groups, so they need two or more.
    for name, minimum in (("generator", 2), ("critic", 1)):
        channels = cfg["model"][name].get("channels")
        if channels is not None and (
            not isinstance(channels, list)
            or not channels
            or any(type(value) is not int or value < minimum for value in channels)
        ):
            raise ValueError(
                f"model.{name}.channels must list integers of at least {minimum}."
            )
    channels = cfg["model"]["generator"].get("channels")
    if channels and any(value % channels[0] for value in channels):
        raise ValueError(
            "model.generator.channels must be integer multiples of its first value."
        )
    steps = cfg["model"]["diffusion"].get("num_steps")
    if steps is not None:
        check_integer(steps, "model.diffusion.num_steps")
    if cfg["model"]["diffusion"]["time_embedding"] not in ("index", "scaled"):
        raise ValueError("model.diffusion.time_embedding must be index or scaled.")
    augmentation = cfg["augmentation"]
    auto_planes = augmentation.get("auto_planes", False)
    check_boolean(auto_planes, "augmentation.auto_planes")
    if auto_planes:
        if "planes" in augmentation:
            raise ValueError("choose augmentation.auto_planes or explicit planes.")
        # Store resolved policies so checkpoint behavior is independent of defaults.
        augmentation["planes"] = auto_plane_policies(cfg)
        augmentation.pop("auto_planes")
    if cfg["model"]["critic"]["input_mode"] not in ("pair", "single"):
        raise ValueError("model.critic.input_mode must be pair or single.")
    get_schedule_steps(cfg["loss"]["group_statistics"], "loss.group_statistics")
    if stage == "low_res":
        profile = cfg["conditioning"]["spatial_profile"]
        if profile["critic_enabled"] and not profile["enabled"]:
            raise ValueError(
                "spatial_profile.critic_enabled requires an enabled profile."
            )
        if profile["enabled"] and not cfg["conditioning"]["height_enabled"]:
            raise ValueError("spatial_profile requires height_enabled.")
    optim = cfg["optim"]
    if "critic_lr" not in optim and "generator_lr" in optim:
        optim["critic_lr"] = optim["generator_lr"]
    validate_training_values(cfg)
    return cfg


def auto_plane_policies(cfg: Mapping) -> dict:
    height = cfg["conditioning"]["height_enabled"]
    groups = cfg["model"]["critic"].get("plane_groups") or ()
    # Rotating xz/yz swaps z with x or y. That hides a height gradient, and hides
    # z anisotropy whenever xy is judged apart from the side planes.
    fixed_z = height or not any("xy" in group and len(group) > 1 for group in groups)
    return {
        plane: {
            "flip_axes": [axis for axis in axes if not (height and axis == "z")],
            "rotate_90": not (fixed_z and "z" in axes),
        }
        for plane, axes in PLANE_DIRECTIONS.items()
    }


def validate_training_settings(cfg: Mapping) -> None:
    """Check settings used only by training, before data or models are built."""
    _require_settings(cfg, TRAINING_SETTINGS)
    train = cfg["train"]
    conditioning, _ = get_trainer_sections(cfg)
    if (
        train["volume_batch_size"] > train["real_batch_size"]
        and conditioning["height_enabled"]
    ):
        raise ValueError(
            "height conditioning requires real_batch_size >= volume_batch_size."
        )


def _require_settings(cfg: Mapping, paths: Sequence[str]) -> None:
    for path in paths:
        target = cfg
        for key in path.split("."):
            if not isinstance(target, Mapping) or key not in target:
                raise ValueError(f"missing required setting for training: {path}")
            target = target[key]


def _setting(cfg: Mapping, path: str):
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


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
    get_sr_resolution(cfg)  # Requires data.hi_res_size.
    groups = get_plane_groups(cfg)
    cfg["model"]["critic"]["plane_groups"] = [
        [PLANES[axis] for axis in axes] for axes in groups.values()
    ]
    validate_training_settings(cfg)
    _require_settings(cfg, ("lr_bank.samples_per_domain",))
    check_integer(cfg["lr_bank"]["samples_per_domain"], "lr_bank.samples_per_domain")
    check_integer(
        cfg["lr_bank"]["refresh_every_steps"], "lr_bank.refresh_every_steps", 0
    )
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
        check_integer(value, f"{name}.{field}", 0)
    return start, ramp
