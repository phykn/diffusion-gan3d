import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.config.data import get_domains, get_plane_groups, get_sizes
from src.config.defaults import STAGE_DEFAULTS, TRAIN_DEFAULTS
from src.config.files import PROJECT_ROOT, load_yaml, prepare_yaml
from src.config.schema import validate_config_keys
from src.config.values import validate_training_values
from src.data.split import resolve_split
from src.plane import PLANE_DIRECTIONS, PLANES


def load_train_config(
    path: str | Path, stage: str = "low_res", data: str | Path | None = None
) -> dict:
    cfg = load_yaml(path)
    if data is not None:
        cfg["data"] = str(data)
    if "data" not in cfg:
        raise ValueError(
            "training config must select a data file or embed data settings."
        )
    if stage == "sr" and "guidance" not in cfg.get("lr_bank", {}):
        raise ValueError("SR config must set lr_bank.guidance explicitly.")
    external_data = isinstance(cfg.get("data"), str)
    if external_data:
        data_path = Path(cfg["data"]).expanduser()
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        cfg["data"] = load_yaml(data_path)
    cfg = normalize_train_config(cfg, stage)
    if external_data:
        for folders in get_domains(cfg["data"]).values():
            for paths in folders.values():
                if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
                    raise TypeError("plane folders must be a sequence of paths.")
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
    for path, value in (TRAIN_DEFAULTS | STAGE_DEFAULTS[stage]).items():
        target = cfg
        *sections, key = path.split(".")
        for section in sections:
            target = target.setdefault(section, {})
        if key not in target:
            target[key] = prepare_yaml(value)
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
    if type(cfg["conditioning"]["height_enabled"]) is not bool:
        raise ValueError("conditioning.height_enabled must be a boolean.")
    augmentation = cfg.get("augmentation", {})
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
    if (
        type(cfg["train"]["structure_every_steps"]) is not int
        or cfg["train"]["structure_every_steps"] < 0
    ):
        raise ValueError("train.structure_every_steps must be a non-negative integer.")
    if (
        type(cfg["model"]["critic"]["pyramid_min_size"]) is not int
        or cfg["model"]["critic"]["pyramid_min_size"] < 2
    ):
        raise ValueError(
            "model.critic.pyramid_min_size must be an integer of at least two."
        )
    if stage == "low_res":
        profile = cfg["conditioning"]["spatial_profile"]
        if type(profile["critic_enabled"]) is not bool or (
            profile["critic_enabled"] and not profile["enabled"]
        ):
            raise ValueError(
                "spatial_profile.critic_enabled requires an enabled profile."
            )
        if type(profile["enabled"]) is not bool:
            raise ValueError("spatial_profile.enabled must be a boolean.")
        if type(profile["num_bins"]) is not int or profile["num_bins"] < 1:
            raise ValueError("spatial_profile.num_bins must be a positive integer.")
        if profile["enabled"] and not cfg["conditioning"]["height_enabled"]:
            raise ValueError("spatial_profile requires height_enabled.")
    optim = cfg["optim"]
    if "critic_lr" not in optim and "generator_lr" in optim:
        optim["critic_lr"] = optim["generator_lr"]
    validate_training_values(cfg)
    return cfg


def validate_sr_config(cfg: dict) -> dict:
    cfg = normalize_train_config(cfg, "sr")
    data = cfg["data"]
    data["crop_size"], data["lo_res_size"], data["hi_res_size"] = get_sr_sizes(cfg)
    groups = get_plane_groups(cfg)
    cfg["model"]["critic"]["plane_groups"] = [
        [PLANES[axis] for axis in axes] for axes in groups.values()
    ]
    for name in (
        "total_steps",
        "volume_batch_size",
        "real_batch_size",
        "slice_pairs_per_plane",
        "weights_every_steps",
    ):
        value = cfg["train"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"train.{name} must be a positive integer.")
    archive = cfg["train"]["archive_every_steps"]
    if archive is not None and (type(archive) is not int or archive < 1):
        raise ValueError(
            "train.archive_every_steps must be a positive integer or null."
        )
    count = cfg["lr_bank"]["samples_per_domain"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("lr_bank.samples_per_domain must be a positive integer.")
    guidance = cfg["lr_bank"]["guidance"]
    refresh = cfg["lr_bank"]["refresh_every_steps"]
    if type(refresh) is not int or refresh < 0:
        raise ValueError("lr_bank.refresh_every_steps must be a non-negative integer.")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
    ):
        raise ValueError("lr_bank.guidance must be finite.")
    return cfg


def get_sr_sizes(cfg: Mapping) -> tuple[int, int, int]:
    cfg = normalize_train_config(cfg, "sr")
    if "hi_res_size" not in cfg["data"]:
        raise ValueError("SR requires data.hi_res_size.")
    return get_sizes(cfg["data"])


def get_schedule_steps(
    section: dict,
    name: str,
) -> tuple[int, int]:
    start = section["start_step"]
    ramp = section["ramp_steps"]
    for field, value in (("start_step", start), ("ramp_steps", ramp)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name}.{field} must be a non-negative integer.")
    return start, ramp
