from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from src.axes import PLANES, validate_num_phases
from src.config.build import build_config, freeze
from src.config.io import PROJECT_ROOT, load_yaml, resolve_path, shorten_path
from src.config.schema import STAGES
from src.data.height import measure_height
from src.data.images import infer_num_phases, load_images
from src.data.profile import measure_depth_profile

SAMPLES_DIR = PROJECT_ROOT / "config" / "samples"
TEMPLATE = "config/samples/default.yaml"
SAMPLE_KEYS = ("images", "structure", "tune", "num_phases", "resolution")
REQUIRED_KEYS = ("images", "resolution", "structure")
RESOLUTION_KEYS = ("crop_size", "lo_res_size", "sr_scale")
STRUCTURE_KEYS = ("isotropic", "depth_gradient", "mirror_symmetric")
SECTION_KEYS = {
    "resolution": RESOLUTION_KEYS,
    "structure": STRUCTURE_KEYS,
    "tune": STAGES,
}

# smooth_l1 is quadratic at these profile errors, so a weight of 1.0 barely pulls.
PROFILE_LOSS_WEIGHT = 30.0
# The critics accept a sheared volume, so its mirrored diagonals must agree.
MIRROR_LOSS_WEIGHT = 10.0
# Flipping half the time hides the tilt from the critic; rarer flips keep it visible.
TILTED_FLIP_PROBABILITY = 0.2


def find_sample(sample):
    path = Path(sample).expanduser()
    if path.suffix not in (".yaml", ".yml"):
        path = SAMPLES_DIR / f"{sample}.yaml"
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in SAMPLES_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"sample not found: {path} (available: {names})")
    return path


def load_sample(sample):
    path = find_sample(sample)
    values = load_yaml(path)
    unknown = set(values) - set(SAMPLE_KEYS)
    if unknown:
        raise ValueError(f"unknown sample setting: {', '.join(sorted(unknown))}")
    missing = [key for key in REQUIRED_KEYS if key not in values]
    if missing:
        raise ValueError(f"{path.name} needs {' and '.join(missing)}; see {TEMPLATE}.")
    _check_sections(values)
    _check_resolution(values["resolution"])
    for key in STRUCTURE_KEYS:
        if type(values["structure"].get(key)) is not bool:
            raise ValueError(f"structure.{key} must be true or false; see {TEMPLATE}.")
    if values.get("num_phases") is not None:
        validate_num_phases(values["num_phases"])
    values["images"] = _list_image_folders(values["images"])
    values["name"] = path.stem
    return values


def build_stage_configs(sample):
    folders = {
        plane: [resolve_path(folder) for folder in plane_folders]
        for plane, plane_folders in sample["images"].items()
    }
    num_phases = sample.get("num_phases")
    images = load_images(folders, num_phases)
    if num_phases is None:
        num_phases = infer_num_phases(images)
    configs = {}
    for stage in STAGES:
        config = build_config(_build_stage_values(sample, stage, num_phases))
        configs[stage] = _add_measurements(config, images)
    return configs


def _check_sections(values):
    for name, allowed in SECTION_KEYS.items():
        if values.get(name) is None:
            values[name] = {}
        section = values[name]
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} must be a mapping.")
        unknown = set(section) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} setting: {', '.join(sorted(unknown))}")
    for stage, tune in values["tune"].items():
        if tune is not None and not isinstance(tune, Mapping):
            raise ValueError(f"tune.{stage} must be a mapping.")


def _check_resolution(resolution):
    for key in RESOLUTION_KEYS:
        if key not in resolution:
            raise ValueError(f"resolution.{key} is required; see {TEMPLATE}.")
    scale = resolution["sr_scale"]
    if type(scale) is not int or scale < 2:
        raise ValueError("resolution.sr_scale must be an integer of at least 2.")
    for key in ("crop_size", "lo_res_size"):
        if type(resolution[key]) is not int or resolution[key] < 1:
            raise ValueError(f"resolution.{key} must be a positive integer.")


def _list_image_folders(images):
    if not isinstance(images, Mapping) or not images:
        raise ValueError(f"images must map xy, xz and yz to folders; see {TEMPLATE}.")
    if set(images) - set(PLANES):
        raise ValueError("images planes must be xy, xz or yz.")
    folders_by_plane = {}
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
        folders_by_plane[plane] = [shorten_path(folder) for folder in folders]
    return folders_by_plane


def _build_stage_values(sample, stage, num_phases):
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
    planes = list(sample["images"])
    _merge(values, _expand_structure(sample["structure"], planes, stage))
    _merge(values, sample["tune"].get(stage) or {})
    return values


def _expand_structure(structure, planes, stage):
    if structure["isotropic"]:
        plane_groups = [planes]
    else:
        # Only z differs: xy sections are judged apart from the xz/yz sections.
        top = [plane for plane in planes if plane == "xy"]
        sides = [plane for plane in planes if plane != "xy"]
        plane_groups = [group for group in (top, sides) if group]
    values = {"model": {"critic": {"plane_groups": plane_groups}}}
    if structure["depth_gradient"]:
        values["conditioning"] = {"height_enabled": True}
        if stage == "lr":
            values["conditioning"]["spatial_profile"] = {"enabled": True}
            values["loss"] = {"spatial_profile_weight": PROFILE_LOSS_WEIGHT}
    if structure["mirror_symmetric"]:
        # SR follows the LR volume through downsample consistency.
        if stage == "lr":
            values.setdefault("loss", {})["mirror_balance"] = {
                "weight": MIRROR_LOSS_WEIGHT
            }
    else:
        values["augmentation"] = {"probability": TILTED_FLIP_PROBABILITY}
        if stage == "sr":
            # A flipped LR volume has the opposite tilt; SR must keep LR's own.
            values["lr_bank"] = {"flips": False}
    return values


def _merge(values, overrides):
    for key, override in overrides.items():
        if isinstance(override, Mapping) and isinstance(values.get(key), dict):
            _merge(values[key], override)
        else:
            values[key] = override


def _add_measurements(config, images):
    measured = {}
    if config.conditioning.height_enabled:
        measured["height_extent"] = measure_height(images)
    if config.profile_enabled:
        measured["depth_profile"] = freeze(
            measure_depth_profile(
                images,
                config.data.num_phases,
                config.conditioning.spatial_profile.num_bins,
            )
        )
    if not measured:
        return config
    return replace(config, data=replace(config.data, **measured))
