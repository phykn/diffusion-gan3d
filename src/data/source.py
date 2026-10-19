from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config.data import get_domains
from src.data.dataset import read_labels
from src.prepare.profile import rebin_profile

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}
ImageGroups = dict[int, dict[int, tuple[tuple[Path, ...], ...]]]


def collect_image_groups(data: dict) -> ImageGroups:
    split = data.get("split", {})
    validation_files = set(split.get("validation_files", []))
    configured = validation_files | set(split.get("validation_regions", {}))
    sources = set()
    domains = {}
    for domain, folders in get_domains(data).items():
        planes = {}
        for axis in sorted(folders):
            values = folders[axis]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"axis {axis} folders must be a sequence of paths.")
            if not values:
                raise ValueError(f"axis {axis} folders must not be empty.")
            paths = tuple(Path(value) for value in values)
            if len({path.resolve() for path in paths}) != len(paths):
                raise ValueError(f"axis {axis} folders must not contain duplicates.")

            groups = []
            for folder in paths:
                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"axis {axis} folder does not exist: {folder}"
                    )
                images = sorted(
                    path
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not images:
                    raise ValueError(f"axis {axis} folder contains no images: {folder}")
                resolved = {path: str(path.resolve()) for path in images}
                sources.update(resolved.values())
                training = tuple(
                    path for path in images if resolved[path] not in validation_files
                )
                if training:
                    groups.append(training)
            if not groups:
                raise ValueError(f"axis {axis} has no training images after splitting.")
            planes[axis] = tuple(groups)
        domains[domain] = planes
    if not configured <= sources:
        raise ValueError("split paths must identify images in data.domains.")
    return domains


def fill_height_extents(cfg: dict) -> dict:
    if cfg["conditioning"]["height_enabled"] and "height_extents" not in cfg["data"]:
        cfg["data"]["height_extents"] = infer_height_extents(cfg["data"])
    return cfg


def fill_depth_profiles(cfg: dict) -> dict:
    profile = cfg["conditioning"].get("spatial_profile", {})
    if (
        cfg.get("stage") == "low_res"
        and profile.get("enabled", False)
        and "depth_profiles" not in cfg["data"]
    ):
        cfg["data"]["depth_profiles"] = infer_depth_profiles(cfg["data"])
    return cfg


def infer_depth_profiles(data: dict, bins: int = 16) -> dict[int, dict | None]:
    """Mean phase fractions of full-height side images in `bins` equal z bands.

    Height-only generation flattens strong depth gradients, so inference uses
    this training-data profile when no vf or vf_profile is requested.
    """
    phases = data["num_phases"]
    profiles = {}
    for domain, planes in collect_image_groups(data).items():
        images = [
            read_labels(path)[0]
            for axis, groups in planes.items()
            if axis != 0
            for paths in groups
            for path in paths
        ]
        if len({len(image) for image in images}) != 1:
            profiles[domain] = None
            continue
        rows = np.mean(
            [np.stack([(img == p).mean(1) for p in range(phases)]) for img in images],
            axis=0,
        )
        bands = rebin_profile(torch.from_numpy(rows)[None], bins)[0].T.numpy()
        values = []
        for band in bands:
            row = [round(float(v), 4) for v in band]
            # Keep the rounded fractions on the simplex that vf_profile requires.
            top = row.index(max(row))
            row[top] = round(1.0 - sum(row) + row[top], 4)
            values.append(row)
        values.append(values[-1])
        profiles[domain] = {
            "axis": "z",
            "interpolation": "constant",
            "points": [[i / bins, row] for i, row in enumerate(values)],
        }
    return profiles


def infer_height_extents(data: dict) -> dict[int, int | None]:
    extents = {}
    for domain, planes in collect_image_groups(data).items():
        sizes = set()
        for axis, groups in planes.items():
            if axis == 0:
                continue
            for paths in groups:
                for path in paths:
                    with Image.open(path) as image:
                        sizes.add(image.height)
        if not sizes:
            raise ValueError(
                "height conditioning requires full-height side images per domain."
            )
        extents[domain] = sizes.pop() if len(sizes) == 1 else None
    if data.get("height_extents", extents) != extents:
        raise ValueError("side-image height differs from saved height_extents.")
    return extents
