from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.phase import validate_num_phases
from src.plane import PLANES
from src.storage import file_hash

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def list_images(folder: Path) -> tuple[Path, ...]:
    """Label images directly in folder; subfolders and other files are skipped."""
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder does not exist: {folder}")
    images = tuple(
        sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )
    if not images:
        raise ValueError(f"image folder contains no images: {folder}")
    return images


def read_labels(path: Path) -> np.ndarray:
    """Return a read-only 2D uint8 label image."""
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"training images must contain one 2D frame: {path}")
            values = np.asarray(image)
    except OSError as exc:
        raise ValueError(f"cannot read image {path}: {exc}") from exc
    if values.ndim != 2:
        raise ValueError(f"training images must be two-dimensional: {path}")
    if values.dtype != np.uint8:
        raise ValueError(f"training images must use uint8 labels: {path}")
    values = np.ascontiguousarray(values)
    values.flags.writeable = False
    return values


@dataclass(frozen=True)
class PlaneImages:
    """Decoded images of one plane, grouped by folder (one group per folder)."""

    plane: str
    folders: tuple[Path, ...]
    paths: tuple[tuple[Path, ...], ...]
    images: tuple[tuple[np.ndarray, ...], ...]

    def all_images(self) -> list[np.ndarray]:
        return [image for group in self.images for image in group]


def find_images(
    images: Mapping[str, Sequence[str | Path]],
) -> dict[str, tuple[tuple[Path, ...], ...]]:
    """Image paths of each plane, one tuple per folder, in plane order."""
    result = {}
    for plane in PLANES:
        if plane not in images:
            continue
        folders = [Path(folder) for folder in images[plane]]
        if len({folder.resolve() for folder in folders}) != len(folders):
            raise ValueError(f"{plane} folders must not contain duplicates.")
        result[plane] = tuple(list_images(folder) for folder in folders)
    return result


def load_images(
    images: Mapping[str, Sequence[str | Path]], num_phases: int | None = None
) -> dict[str, PlaneImages]:
    """Decode every image once; labels must be below num_phases when given."""
    planes = {}
    for plane, groups in find_images(images).items():
        decoded = []
        for group in groups:
            values = []
            for path in group:
                image = read_labels(path)
                if num_phases is not None:
                    largest = int(image.max())
                    if largest >= num_phases:
                        raise ValueError(
                            f"phase labels must be in [0, {num_phases - 1}], "
                            f"found {largest}: {path}"
                        )
                values.append(image)
            decoded.append(tuple(values))
        planes[plane] = PlaneImages(
            plane=plane,
            folders=tuple(Path(folder) for folder in images[plane]),
            paths=groups,
            images=tuple(decoded),
        )
    return planes


def infer_num_phases(planes: Mapping[str, PlaneImages]) -> int:
    """Count phases as the largest label in the training images plus one.

    Every smaller label must appear too: 0/255 masks, or 2- and 4-bit PNGs that
    PIL scales to 0..255, would otherwise make hundreds of phases.
    """
    present = np.zeros(256, dtype=bool)
    largest, source = -1, None
    for images in planes.values():
        for group, paths in zip(images.images, images.paths, strict=True):
            for image, path in zip(group, paths, strict=True):
                present |= np.bincount(image.ravel(), minlength=256) > 0
                if int(image.max()) > largest:
                    largest, source = int(image.max()), path
    count = largest + 1
    missing = np.flatnonzero(~present[:count])
    if missing.size:
        raise ValueError(
            f"label {missing[0]} never appears below the largest label {count - 1}; "
            "use labels 0..K-1, or set num_phases to keep an unused phase. "
            f"Largest label in: {source}"
        )
    return validate_num_phases(count)


def fingerprint(planes: Mapping[str, PlaneImages]) -> dict[str, dict[str, str]]:
    """sha256 of every image, keyed by plane and folder-relative file name."""
    result = {}
    for plane, images in planes.items():
        entries = {}
        for index, (folder, paths) in enumerate(
            zip(images.folders, images.paths, strict=True)
        ):
            for path in paths:
                name = path.relative_to(folder).as_posix()
                key = name if len(images.folders) == 1 else f"{index}/{name}"
                entries[key] = file_hash(path)
        result[plane] = entries
    return result
