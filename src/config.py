import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.plane import PLANES, get_axis
from src.prepare.resize import scaled_size

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Advanced options may be omitted from presets. Resolved runs save these values.
TRAIN_DEFAULTS = {
    "optim.ema_decay": 0.999,
    "train.mixed_precision": True,
    "train.structure_every_steps": 100,
    "model.critic.pyramid_min_size": 16,
    "conditioning.height_enabled": False,
}
STAGE_DEFAULTS = {
    "low_res": {
        "model.gradient_checkpointing": True,
        "model.generator.embedding_channels": 128,
        "model.generator.latent_channels": 64,
        "model.generator.anchor_multiscale_input": True,
        "model.diffusion.beta_min": 0.1,
        "model.diffusion.beta_max": 20.0,
        "model.diffusion.time_embedding": "index",
        "conditioning.domain_keep_probability": 0.8,
        "conditioning.dropout_probability_per_case": 0.05,
        "conditioning.anchor.start_step": 0,
        "conditioning.anchor.ramp_steps": 500,
        "conditioning.anchor.borrowed_plane_probability": 0.2,
        "conditioning.anchor.bank_capacity": 4,
        "conditioning.anchor.plane_spacing": 16,
        "loss.critic_local_weight": 0.5,
        "loss.r1_every_steps": 16,
        "loss.r2_weight": 0.0,
        "loss.connectivity.start_step": 0,
        "loss.connectivity.ramp_steps": 20000,
        "loss.connectivity.windows_per_plane": 4,
        "optim.adam_betas": [0.5, 0.9],
        "train.initial_weights": None,
        "train.num_workers": 0,
    },
    "sr": {
        "model.generator.noise_channels": 1,
        "loss.downsample_mse_tolerance": 0.005,
        "optim.adam_betas": [0.0, 0.9],
        "lr_bank.refresh_every_steps": 1000,
        "conditioning.coarse_corruption_probability": 0.5,
        "conditioning.coarse_corruption_strength": 0.2,
    },
}


def validate_config_keys(cfg: Mapping, stage: str) -> None:
    """Reject misspellings at input boundaries, including obsolete configuration keys."""
    common = {
        "stage": None,
        "data": dict.fromkeys(
            (
                "domains",
                "num_phases",
                "crop_size",
                "lo_res_size",
                "thickness_axis",
                "height_extents",
            )
        ),
        "augmentation": {
            "probability": None,
            "planes": {
                plane: dict.fromkeys(("flip_axes", "rotate_90")) for plane in PLANES
            },
        },
        "optim": dict.fromkeys(
            ("generator_lr", "critic_lr", "adam_betas", "ema_decay")
        ),
    }
    if stage == "low_res":
        schema = common | {
            "model": {
                "gradient_checkpointing": None,
                "generator": dict.fromkeys(
                    (
                        "channels",
                        "embedding_channels",
                        "latent_channels",
                        "anchor_multiscale_input",
                    )
                ),
                "critic": dict.fromkeys(
                    ("channels", "plane_groups", "pyramid_min_size")
                ),
                "diffusion": dict.fromkeys(
                    ("num_steps", "beta_min", "beta_max", "time_embedding")
                ),
            },
            "conditioning": {
                "height_enabled": None,
                "domain_keep_probability": None,
                "dropout_probability_per_case": None,
                "anchor": dict.fromkeys(
                    (
                        "probability",
                        "start_step",
                        "ramp_steps",
                        "borrowed_plane_probability",
                        "bank_capacity",
                        "plane_spacing",
                    )
                ),
            },
            "loss": dict.fromkeys(
                (
                    "critic_local_weight",
                    "r1_weight",
                    "r1_every_steps",
                    "r2_weight",
                    "anchor_pixel_weight",
                    "volume_fraction_weight",
                )
            )
            | {
                "connectivity": dict.fromkeys(
                    (
                        "max_slice_gap",
                        "adversarial_weight",
                        "normal_transition_weight",
                        "start_step",
                        "ramp_steps",
                        "windows_per_plane",
                    )
                ),
            },
            "train": dict.fromkeys(
                (
                    "total_steps",
                    "mixed_precision",
                    "initial_weights",
                    "num_workers",
                    "real_batch_size",
                    "volume_batch_size",
                    "slice_pairs_per_plane",
                    "weights_every_steps",
                    "archive_every_steps",
                    "structure_every_steps",
                )
            ),
        }
    else:
        schema = common | {
            "model": {
                "generator": dict.fromkeys(
                    ("channels", "blocks", "noise_channels", "scale_factor")
                ),
                "critic": dict.fromkeys(
                    ("channels", "plane_groups", "pyramid_min_size")
                ),
            },
            "conditioning": dict.fromkeys(
                (
                    "coarse_corruption_probability",
                    "coarse_corruption_strength",
                    "height_enabled",
                )
            ),
            "loss": dict.fromkeys(
                (
                    "gradient_penalty_weight",
                    "downsample_consistency_weight",
                    "downsample_mse_tolerance",
                )
            ),
            "lr_bank": dict.fromkeys(
                ("samples_per_domain", "guidance", "refresh_every_steps")
            ),
            "source": dict.fromkeys(
                ("weights", "weights_sha256", "config_sha256", "bank", "bank_sha256")
            ),
            "train": dict.fromkeys(
                (
                    "total_steps",
                    "mixed_precision",
                    "volume_batch_size",
                    "slices_per_plane",
                    "critic_updates_per_step",
                    "checkpoint_every_steps",
                    "structure_every_steps",
                )
            ),
        }

    def check(values, allowed, path=""):
        if not isinstance(values, Mapping):
            raise TypeError(f"{path or 'config'} must be a mapping.")
        for key, value in values.items():
            name = f"{path}.{key}" if path else str(key)
            if key not in allowed:
                raise ValueError(f"unknown training setting: {name}")
            if allowed[key] is not None:
                check(value, allowed[key], name)

    check(cfg, schema)
    domains = cfg.get("data", {}).get("domains", {})
    if not isinstance(domains, Mapping):
        raise TypeError("data.domains must be a mapping.")
    for planes in domains.values():
        if not isinstance(planes, Mapping):
            raise TypeError("data.domains must map domains to plane folders.")
        if any(plane not in PLANES for plane in planes):
            raise ValueError("data.domains must use plane names xy, xz or yz.")


def load_train_config(
    path: str | Path, stage: str = "low_res", data: str | Path | None = None
) -> dict:
    """Resolve one data file, then return a self-contained training configuration."""
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
    """Validate the current schema and resolve omitted options without mutating input."""
    if stage not in ("low_res", "sr") or cfg.get("stage", stage) != stage:
        raise ValueError(f"expected stage: {stage}.")
    validate_config_keys(cfg, stage)
    cfg = prepare_yaml(cfg)
    cfg["stage"] = stage
    for path, value in (TRAIN_DEFAULTS | STAGE_DEFAULTS[stage]).items():
        target = cfg
        *sections, key = path.split(".")
        for section in sections:
            target = target.setdefault(section, {})
        if key not in target:
            target[key] = prepare_yaml(value)
    if type(cfg["conditioning"]["height_enabled"]) is not bool:
        raise ValueError("conditioning.height_enabled must be a boolean.")
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
    optim = cfg["optim"]
    if "critic_lr" not in optim and "generator_lr" in optim:
        optim["critic_lr"] = optim["generator_lr"]
    return cfg


def validate_sr_source(data: Mapping, base_data: Mapping) -> None:
    """Require the same field of view and label/domain contract, not image paths."""
    if "lo_res_size" not in base_data:
        raise ValueError("SR requires stage-1 weights trained with lo_res_size.")
    if (data["crop_size"], data["lo_res_size"]) != (
        base_data["crop_size"],
        base_data["lo_res_size"],
    ):
        raise ValueError("SR data resolution must match the stage-1 crop/LR sizes.")
    if data["num_phases"] != base_data["num_phases"]:
        raise ValueError("SR data.num_phases must match the stage-1 model.")
    if set(get_domains(data)) != set(get_domains(base_data)):
        raise ValueError("SR data domain IDs must match the stage-1 model.")
    if data.get("height_extents") != base_data.get("height_extents") or (
        data.get("height_extents")
        and data.get("thickness_axis") != base_data.get("thickness_axis")
    ):
        raise ValueError("SR height coordinates must match the stage-1 model.")


def get_sizes(
    data: Mapping[str, object], scale_factor: float | None = None
) -> tuple[int, int, int]:
    """Return original crop, LR grid and the independently selected target size."""
    if any(
        key in data
        for key in (
            "input_size",
            "scale_factor",
            "hi_res_size",
            "allow_part",
            "allow_partial_crops",
        )
    ):
        raise ValueError(
            "use crop_size and lo_res_size; SR scale_factor belongs to model.generator."
        )
    crop, low = data["crop_size"], data["lo_res_size"]
    for name, size in (("crop_size", crop), ("lo_res_size", low)):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{name} must be a positive integer.")
    high = scaled_size(low, 1 if scale_factor is None else scale_factor)
    return crop, low, high


def get_sr_sizes(cfg: Mapping) -> tuple[int, int, int]:
    cfg = normalize_train_config(cfg, "sr")
    generator = cfg.get("model", {}).get("generator", {})
    if generator.get("scale_factor") is None:
        raise ValueError("SR requires model.generator.scale_factor.")
    return get_sizes(cfg["data"], generator["scale_factor"])


def get_plane_groups(cfg: Mapping) -> dict[str, tuple[int, ...]]:
    """Resolve disjoint critic groups over the union of observed planes."""
    active = {axis for axes in get_domains(cfg["data"]).values() for axis in axes}
    groups = cfg["model"]["critic"].get("plane_groups")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("model.critic.plane_groups must be a non-empty list of lists.")
    result = {}
    seen = set()
    for group in groups:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("each critic plane group must be a non-empty list.")
        if any(not isinstance(plane, str) or plane not in PLANES for plane in group):
            raise ValueError("critic plane groups must use xy, xz and yz names.")
        axes = tuple(sorted(get_axis(plane) for plane in group))
        if len(set(axes)) != len(axes) or seen.intersection(axes):
            raise ValueError("each plane must occur in exactly one critic group.")
        seen.update(axes)
        result["_".join(PLANES[axis] for axis in axes)] = axes
    if seen != active:
        raise ValueError("critic plane groups must cover exactly the observed planes.")
    return dict(sorted(result.items(), key=lambda item: item[1]))


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
        axes = {}
        for plane, paths in folders.items():
            if plane not in PLANES:
                raise ValueError("data.domains must use plane names xy, xz or yz.")
            axis = get_axis(plane)
            if axis in axes:
                raise ValueError(f"domain {domain} contains duplicate plane {plane!r}.")
            axes[axis] = paths
        parsed[domain] = axes
    return parsed


@dataclass(frozen=True)
class GenerationSettings:
    guidance: float = 1.0
    anchor_strength: float = 1.0
    overlap: int = 8


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(Path(__file__).resolve().parents[1] / "config" / "gen.yaml")
    unknown = set(section) - {
        "guidance",
        "anchor_strength",
        "overlap",
    }
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown generation setting: {names}")

    guidance = float(section.get("guidance", 1.0))
    anchor_strength = section.get("anchor_strength", 1.0)
    if (
        not isinstance(anchor_strength, (int, float))
        or isinstance(anchor_strength, bool)
        or not math.isfinite(anchor_strength)
        or not 0.0 <= anchor_strength <= 1.0
    ):
        raise ValueError("anchor_strength must be between zero and one.")
    overlap = section.get("overlap", 8)
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    return GenerationSettings(
        guidance=guidance,
        anchor_strength=float(anchor_strength),
        overlap=overlap,
    )


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


def find_train_config(weights: str | Path) -> Path:
    path = Path(weights).resolve()
    for parent in path.parents:
        config = parent / "train.yaml"
        if config.is_file():
            return config
    raise FileNotFoundError(f"train.yaml was not found above weights file: {path}")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping.")
    return data


def save_yaml(path: str | Path, data: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        blocks = [
            yaml.dump({key: value}, Dumper=ConfigDumper, sort_keys=False)
            for key, value in prepare_yaml(data).items()
        ]
        file.write("\n".join(blocks) if blocks else "{}\n")


class ConfigDumper(yaml.SafeDumper):
    def represent_list(self, values):
        return self.represent_sequence("tag:yaml.org,2002:seq", values, flow_style=True)


ConfigDumper.add_representer(list, ConfigDumper.represent_list)


def prepare_yaml(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: prepare_yaml(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [prepare_yaml(item) for item in value]
    return value
