from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from src.config.data import get_domains

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


def infer_height_extents(data: dict) -> dict[int, int | None]:
    if data.get("thickness_axis") not in (None, "z"):
        raise ValueError("height conditioning uses the fixed z axis.")
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
                "height conditioning requires full-thickness side images per domain."
            )
        extents[domain] = sizes.pop() if len(sizes) == 1 else None
    if data.get("height_extents", extents) != extents:
        raise ValueError("side-image thickness differs from saved height_extents.")
    return extents
