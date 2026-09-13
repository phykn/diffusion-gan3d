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
        "loss.critic_local_weight": 0.5,
        "loss.r1_every_steps": 16,
        "loss.r2_weight": 0.0,
        "optim.adam_betas": [0.5, 0.9],
        "train.initial_weights": None,
        "train.num_workers": 0,
        "train.stability_version": 1,
    },
    "sr": {
        "model.generator.noise_channels": 1,
        "loss.downsample_mse_tolerance": 0.005,
        "loss.downsample_temperature": 0.05,
        "optim.adam_betas": [0.0, 0.9],
    },
}


def load_train_config(
    path: str | Path, stage: str = "low_res", data: str | Path | None = None
) -> dict:
    """Resolve one data file, then return a self-contained training configuration."""
    cfg = load_yaml(path)
    if data is not None:
        cfg["data"] = str(data)
    if cfg.get("stage") == stage and "data" not in cfg:
        raise ValueError(
            "training config must select a data file or embed data settings."
        )
    if (
        stage == "sr"
        and cfg.get("stage") == "sr"
        and "guidance" not in cfg.get("lr_bank", {})
    ):
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
    """Migrate old names and resolve omitted options without changing explicit values."""
    if stage not in ("low_res", "sr") or cfg.get("stage", stage) != stage:
        raise ValueError(f"expected stage: {stage}.")
    cfg = prepare_yaml(cfg)
    # Old training seeds no longer control initialization or batch sampling.
    cfg.get("train", {}).pop("seed", None)
    cfg["stage"] = stage
    moves = {
        "data.num_phase": "data.num_phases",
        "data.allow_part": "data.allow_partial_crops",
        "data.augment": "augmentation.mode",
        "data.augment_prob": "augmentation.probability",
        "train.steps": "train.total_steps",
        "train.amp": "train.mixed_precision",
    }
    if stage == "low_res":
        moves.update(
            {
                "data.batch_size": "train.real_batch_size",
                "data.num_workers": "train.num_workers",
                "data.domain_prob": "conditioning.domain_keep_probability",
                "model.grad_checkpoint": "model.gradient_checkpointing",
                "model.generator.condition_channels": "model.generator.embedding_channels",
                "anchor.multiscale_input": "model.generator.anchor_multiscale_input",
                "diffusion.steps": "model.diffusion.num_steps",
                "diffusion.beta_min": "model.diffusion.beta_min",
                "diffusion.beta_max": "model.diffusion.beta_max",
                "anchor.train_prob": "conditioning.anchor.probability",
                "anchor.start_step": "conditioning.anchor.start_step",
                "anchor.ramp_steps": "conditioning.anchor.ramp_steps",
                "anchor.cross_domain_prob": "conditioning.anchor.borrowed_plane_probability",
                "condition_dropout.joint_each_prob": "conditioning.dropout_probability_per_case",
                "anchor.pixel_weight": "loss.anchor_pixel_weight",
                "anchor.connectivity.max_gap": "loss.connectivity.max_slice_gap",
                "anchor.connectivity.weight": "loss.connectivity.adversarial_weight",
                "anchor.connectivity.phase_transition_weight": "loss.connectivity.normal_transition_weight",
                "vf.weight": "loss.volume_fraction_weight",
                "model.critic.local_loss_weight": "loss.critic_local_weight",
                "model.critic.r1_weight": "loss.r1_weight",
                "model.critic.r1_interval": "loss.r1_every_steps",
                "train.init_weights": "train.initial_weights",
                "train.pairs_per_axis": "train.slice_pairs_per_plane",
                "train.update_weights_every": "train.weights_every_steps",
                "train.archive_every": "train.archive_every_steps",
            }
        )
    else:
        moves.update(
            {
                "model.channels": "model.generator.channels",
                "model.blocks": "model.generator.blocks",
                "model.noise_channels": "model.generator.noise_channels",
                "data.scale_factor": "model.generator.scale_factor",
                "critic.channels": "model.critic.channels",
                "optim.betas": "optim.adam_betas",
                "train.batch_size": "train.volume_batch_size",
                "train.bank_size": "lr_bank.samples_per_domain",
                "train.slices_per_axis": "train.slices_per_plane",
                "train.critic_steps": "train.critic_updates_per_step",
                "train.save_every": "train.checkpoint_every_steps",
                "train.gradient_penalty": "loss.gradient_penalty_weight",
                "train.lr_weight": "loss.downsample_consistency_weight",
                "train.lr_tolerance": "loss.downsample_mse_tolerance",
                "train.temperature": "loss.downsample_temperature",
            }
        )
    for old, new in moves.items():
        source = cfg
        parents = []
        *sections, key = old.split(".")
        for section in sections:
            if not isinstance(source, Mapping) or section not in source:
                break
            parents.append((source, section))
            source = source[section]
        else:
            if not isinstance(source, Mapping) or key not in source:
                continue
            target = cfg
            *sections, new_key = new.split(".")
            for section in sections:
                target = target.setdefault(section, {})
            if new_key in target:
                raise ValueError(f"use {new}, not both {old} and {new}.")
            target[new_key] = source.pop(key)
            for parent, section in reversed(parents):
                if not parent[section]:
                    del parent[section]
    if stage == "sr":
        # These inherited stage-1 fields were never used by SR.
        for key in ("batch_size", "num_workers", "domain_prob"):
            cfg.get("data", {}).pop(key, None)
    for path, value in (TRAIN_DEFAULTS | STAGE_DEFAULTS[stage]).items():
        target = cfg
        *sections, key = path.split(".")
        for section in sections:
            target = target.setdefault(section, {})
        if key not in target:
            target[key] = prepare_yaml(value)
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


def get_sizes(
    data: Mapping[str, object], scale_factor: float | None = None
) -> tuple[int, int, int]:
    """Return crop, LR and target size; legacy data.scale_factor retains its HR step."""
    crop = data["crop_size"]
    if "lo_res_size" in data:
        if "input_size" in data:
            raise ValueError("use lo_res_size instead of input_size, not both.")
        low = data["lo_res_size"]
        scale = data.get("scale_factor", 1) if scale_factor is None else scale_factor
        high = scaled_size(low, scale)
        if data.get("allow_partial_crops", data.get("allow_part", False)):
            raise ValueError("partial crops are not supported with lo_res_size.")
    else:
        if "scale_factor" in data or scale_factor is not None:
            raise ValueError("scale_factor requires lo_res_size.")
        low = data["input_size"]
        high = low
    for name, size in (("crop_size", crop), ("lo_res_size", low)):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if "hi_res_size" in data and data["hi_res_size"] != high:
        raise ValueError("hi_res_size is derived from lo_res_size * scale_factor.")
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
    if "plane_groups" not in cfg["model"]["critic"]:
        return {PLANES[axis]: (axis,) for axis in sorted(active)}
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
    anchor_spread: float = 0.1


def load_generation_settings() -> GenerationSettings:
    section = load_yaml(Path(__file__).resolve().parents[1] / "config" / "gen.yaml")
    unknown = set(section) - {
        "guidance",
        "anchor_strength",
        "anchor_spread",
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
    anchor_spread = section.get("anchor_spread", 0.1)
    if (
        not isinstance(anchor_spread, (int, float))
        or isinstance(anchor_spread, bool)
        or not math.isfinite(anchor_spread)
        or anchor_spread <= 0.0
    ):
        raise ValueError("anchor_spread must be positive and finite.")
    overlap = section.get("overlap", 8)
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    return GenerationSettings(
        guidance=guidance,
        anchor_strength=float(anchor_strength),
        overlap=overlap,
        anchor_spread=float(anchor_spread),
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
