"""Sample YAML files (config/samples/*.yaml) and their two stage configs."""

import copy
from collections.abc import Mapping
from pathlib import Path

from src.config.io import PROJECT_ROOT, load_yaml, project_path, stored_path
from src.config.schema import (
    STAGES,
    StageConfig,
    build_config,
    to_dict,
)
from src.plane import PLANES

SAMPLES_DIR = PROJECT_ROOT / "config" / "samples"
SAMPLE_KEYS = ("images", "structure", "tune", "num_phases", "resolution")
RESOLUTION = ("crop_size", "lo_res_size", "sr_scale")
STRUCTURE = ("isotropic", "depth_gradient", "mirror_symmetric")
TEMPLATE = "config/samples/default.yaml"


def find_sample(sample: str | Path) -> Path:
    """Resolve a sample name in config/samples/, or a path to a sample YAML."""
    path = Path(sample).expanduser()
    if path.suffix not in (".yaml", ".yml"):
        path = SAMPLES_DIR / f"{sample}.yaml"
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in SAMPLES_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"sample not found: {path} (available: {names})")
    return path


def load_sample(sample: str | Path) -> dict:
    path = find_sample(sample)
    cfg = load_yaml(path)
    unknown = set(cfg) - set(SAMPLE_KEYS)
    if unknown:
        raise ValueError(f"unknown sample setting: {', '.join(sorted(unknown))}")
    missing = [
        name for name in ("images", "resolution", "structure") if name not in cfg
    ]
    if missing:
        raise ValueError(f"{path.name} needs {' and '.join(missing)}; see {TEMPLATE}.")
    for name, allowed in (
        ("resolution", RESOLUTION),
        ("structure", STRUCTURE),
        ("tune", STAGES),
    ):
        section = cfg.setdefault(name, {})
        if section is None:
            section = cfg[name] = {}
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} must be a mapping.")
        unknown = set(section) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} setting: {', '.join(sorted(unknown))}")
    for stage, tune in cfg["tune"].items():
        if tune is not None and not isinstance(tune, Mapping):
            raise ValueError(f"tune.{stage} must be a mapping.")
    for name in RESOLUTION:
        if name not in cfg["resolution"]:
            raise ValueError(f"resolution.{name} is required; see {TEMPLATE}.")
    scale = cfg["resolution"]["sr_scale"]
    if type(scale) is not int or scale < 2:
        raise ValueError("resolution.sr_scale must be an integer of at least 2.")
    for name in ("crop_size", "lo_res_size"):
        value = cfg["resolution"][name]
        if type(value) is not int or value < 1:
            raise ValueError(f"resolution.{name} must be a positive integer.")
    for name in STRUCTURE:
        if type(cfg["structure"].get(name)) is not bool:
            raise ValueError(f"structure.{name} must be true or false; see {TEMPLATE}.")
    cfg["images"] = sample_images(cfg)
    cfg["name"] = path.stem
    return cfg


def sample_images(sample: Mapping) -> dict[str, list[str]]:
    images = sample["images"]
    if not isinstance(images, Mapping) or not images:
        raise ValueError(f"images must map xy, xz and yz to folders; see {TEMPLATE}.")
    if set(images) - set(PLANES):
        raise ValueError("images planes must be xy, xz or yz.")
    result = {}
    for plane in PLANES:
        if plane not in images:
            continue
        folders = images[plane]
        if isinstance(folders, str):
            folders = [folders]
        if (
            not isinstance(folders, list)
            or not folders
            or not all(isinstance(folder, str) for folder in folders)
        ):
            raise ValueError(f"images.{plane} must be a folder or a list of folders.")
        result[plane] = [stored_path(folder) for folder in folders]
    return result


def structure_settings(structure: Mapping, planes: list[str], stage: str) -> dict:
    """The one place the structure flags become stage settings."""
    if structure["isotropic"]:
        groups = [planes]
    else:
        # Only z differs: xy sections are judged apart from the xz/yz sections.
        groups = [[p for p in planes if p == "xy"], [p for p in planes if p != "xy"]]
        groups = [group for group in groups if group]
    cfg = {"model": {"critic": {"plane_groups": groups}}}
    if structure["depth_gradient"]:
        cfg["conditioning"] = {"height_enabled": True}
        if stage == "lr":
            cfg["conditioning"]["spatial_profile"] = {"enabled": True}
            # smooth_l1 (beta=1) is quadratic at these profile errors, so 1.0
            # barely pulls.
            cfg["loss"] = {"spatial_profile_weight": 30.0}
    if structure["mirror_symmetric"]:
        if stage == "lr":
            # A shear such as f(y, x - z) keeps two planes realistic at the cost
            # of one; the critics accept that trade, so mirrored diagonals must
            # agree. SR follows the LR volume through downsample consistency.
            cfg.setdefault("loss", {})["mirror_balance"] = {"weight": 10.0}
    else:
        # With height kept, a side plane's only transform is a flip; at 0.5 the
        # critic cannot tell either tilt from the real one and the generator
        # drifts to one. Rarer flips keep the tilt visible but still fight
        # critic overfitting.
        cfg["augmentation"] = {"probability": 0.2}
        if stage == "sr":
            # A flipped LR volume has the opposite tilt; SR must keep LR's own.
            cfg["lr_bank"] = {"flips": False}
    return cfg


def merge(base: dict, override: Mapping) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def stage_values(sample: Mapping, stage: str, num_phases: int) -> dict:
    """Schema input for one stage: data, then structure flags, then tune."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}.")
    resolution = sample["resolution"]
    values = {
        "stage": stage,
        "nickname": sample["name"],
        "data": {
            "images": sample["images"],
            "num_phases": num_phases,
            "crop_size": resolution["crop_size"],
            "lo_res_size": resolution["lo_res_size"],
            "hi_res_size": resolution["lo_res_size"] * resolution["sr_scale"],
        },
    }
    merge(
        values, structure_settings(sample["structure"], list(sample["images"]), stage)
    )
    merge(values, sample["tune"].get(stage) or {})
    return values


def build_stage_configs(sample: Mapping) -> dict[str, StageConfig]:
    """Both stage configs, with phase count, height extent and depth profile
    measured from the images. Fails before any training on a bad tune."""
    from src.data.images import infer_num_phases, load_images

    folders = {
        plane: [project_path(path) for path in paths]
        for plane, paths in sample["images"].items()
    }
    num_phases = sample.get("num_phases")
    planes = load_images(folders, num_phases)
    if num_phases is None:
        num_phases = infer_num_phases(planes)
    configs = {}
    for stage in STAGES:
        config = build_config(stage_values(sample, stage, num_phases))
        configs[stage] = measure_data(config, planes)
    return configs


def measure_data(config: StageConfig, planes: Mapping) -> StageConfig:
    """Fill data.height_extent and data.depth_profile from the images."""
    from src.data.height import depth_profile, height_extent

    values = {}
    if config.conditioning.height_enabled:
        values["height_extent"] = height_extent(planes)
    if config.profile_enabled:
        values["depth_profile"] = depth_profile(
            planes,
            config.data.num_phases,
            config.conditioning.spatial_profile.num_bins,
        )
    if not values:
        return config
    data = to_dict(config.data) | values
    return build_config(to_dict(config) | {"data": data})
