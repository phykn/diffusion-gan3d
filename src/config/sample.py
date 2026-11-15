from collections.abc import Mapping
from pathlib import Path

from src.axes import PLANES
from src.config.build import build_config, freeze, parse, to_dict
from src.config.io import PROJECT_ROOT, load_yaml, resolve_path, shorten_path
from src.config.schema import STAGES, Sample
from src.data.height import measure_height
from src.data.images import infer_num_phases, load_images
from src.data.profile import measure_depth_profile

SAMPLES_DIR = PROJECT_ROOT / "config" / "samples"
TEMPLATE = "config/samples/default.yaml"

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
    try:
        values = to_dict(parse(Sample, load_yaml(path)))
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}; see {TEMPLATE}.") from None
    images = {}
    for plane in PLANES:
        if plane in values["images"]:
            folders = values["images"][plane]
            folders = [folders] if isinstance(folders, str) else folders
            images[plane] = [shorten_path(folder) for folder in folders]
    values["images"] = images
    values["tune"] = values["tune"] or {}
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
        # SR too: downsample consistency holds only above the LR voxel scale.
        values.setdefault("loss", {})["mirror_balance"] = {"weight": MIRROR_LOSS_WEIGHT}
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
    data = config.data.model_copy(update=measured)
    return config.model_copy(update={"data": data})
