import copy
from collections.abc import Mapping
from pathlib import Path

from src.config.files import PROJECT_ROOT, load_yaml, project_path
from src.config.train import normalize_train_config
from src.data.source import infer_num_phases
from src.plane import PLANES

SAMPLES_DIR = PROJECT_ROOT / "config" / "samples"
BASE_DIR = PROJECT_ROOT / "config" / "base"
# Short stage names for config/base/<name>.yaml, tune keys, and run folders.
STAGE_NAMES = {"low_res": "lr", "sr": "sr"}
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
        ("tune", STAGE_NAMES.values()),
    ):
        section = cfg.setdefault(name, {})
        if not isinstance(section, Mapping):
            raise TypeError(f"{name} must be a mapping.")
        unknown = set(section) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} setting: {', '.join(sorted(unknown))}")
    for name in RESOLUTION:
        if name not in cfg["resolution"]:
            raise ValueError(f"resolution.{name} is required; see {TEMPLATE}.")
    scale = cfg["resolution"]["sr_scale"]
    if type(scale) is not int or scale < 2:
        raise ValueError("resolution.sr_scale must be an integer of at least 2.")
    for name in STRUCTURE:
        if type(cfg["structure"].get(name)) is not bool:
            raise ValueError(f"structure.{name} must be true or false; see {TEMPLATE}.")
    cfg["name"] = path.stem
    return cfg


def build_train_config(sample: Mapping, stage: str) -> dict:
    """Return the stage YAML: base, then structure settings, then tune; later wins."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"stage must be one of {', '.join(STAGE_NAMES)}.")
    base = load_yaml(BASE_DIR / f"{STAGE_NAMES[stage]}.yaml")
    images = get_images(sample)
    num_phases = sample.get("num_phases")
    if num_phases is None:
        folders = {
            plane: [str(project_path(path)) for path in paths]
            for plane, paths in images.items()
        }
        num_phases = infer_num_phases({"domains": {0: folders}})
    resolution = sample["resolution"]
    data = {
        "domains": {0: images},
        "num_phases": num_phases,
        "crop_size": resolution["crop_size"],
        "lo_res_size": resolution["lo_res_size"],
        "hi_res_size": resolution["lo_res_size"] * resolution["sr_scale"],
    }
    cfg = {"stage": stage, "nickname": sample["name"], "data": data} | base
    merge(cfg, get_structure_settings(sample["structure"], list(images), stage))
    merge(cfg, copy.deepcopy(sample["tune"].get(STAGE_NAMES[stage]) or {}))
    # A bad tune.sr fails here, not after the LR stage has trained.
    normalize_train_config(cfg, stage)
    return cfg


def describe_resolution(data: Mapping) -> str:
    """One line on how many source pixels an LR and an HR voxel span."""
    crop = data["crop_size"]
    low, high = data["lo_res_size"], data["hi_res_size"]
    return (
        f"Resolution: crop {crop} px -> LR {low} voxels (1 voxel = {crop / low:g} px), "
        f"HR {high} voxels (1 voxel = {crop / high:g} px)"
    )


def get_images(sample: Mapping) -> dict[str, list[str]]:
    images = sample["images"]
    if not isinstance(images, Mapping) or not images:
        raise ValueError(f"images must map xy, xz and yz to folders; see {TEMPLATE}.")
    if set(images) - set(PLANES):
        raise ValueError("images planes must be xy, xz or yz.")
    return {
        plane: [images[plane]] if isinstance(images[plane], str) else images[plane]
        for plane in PLANES
        if plane in images
    }


def get_structure_settings(structure: Mapping, planes: list[str], stage: str) -> dict:
    if structure["isotropic"]:
        groups = [planes]
    else:
        # Only z differs: xy sections are judged apart from the xz/yz sections.
        groups = [[p for p in planes if p == "xy"], [p for p in planes if p != "xy"]]
        groups = [group for group in groups if group]
    cfg = {"model": {"critic": {"plane_groups": groups}}}
    if structure["depth_gradient"]:
        cfg["conditioning"] = {"height_enabled": True}
        if stage == "low_res":
            cfg["conditioning"]["spatial_profile"] = {"enabled": True}
            # smooth_l1 (beta=1) is quadratic at these profile errors, so 1.0 barely pulls.
            cfg["loss"] = {"spatial_profile_weight": 30.0}
    if structure["mirror_symmetric"]:
        if stage == "low_res":
            # A shear such as f(y, x - z) keeps two planes realistic at the cost
            # of one; the critics accept that trade, so mirrored diagonals must
            # agree. SR follows the LR volume through downsample consistency.
            cfg.setdefault("loss", {})["mirror_balance"] = {"weight": 10.0}
    else:
        # With height kept, a side plane's only transform is a flip; at 0.5 the
        # critic cannot tell either tilt from the real one and the generator drifts
        # to one. Rarer flips keep the tilt visible but still fight critic overfitting.
        cfg["augmentation"] = {"probability": 0.2}
    return cfg


def merge(base: dict, override: Mapping) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
