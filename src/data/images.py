from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image

from src.axes import PLANES, validate_num_phases
from src.storage import file_hash

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def list_images(folder):
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


def read_labels(path):
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
    plane: str
    folders: tuple[Path, ...]
    paths: tuple[tuple[Path, ...], ...]
    images: tuple[tuple[np.ndarray, ...], ...]

    def all_images(self):
        return list(chain.from_iterable(self.images))


def load_images(images, num_phases=None):
    planes = {}
    for plane in PLANES:
        if plane not in images:
            continue
        folders = tuple(map(Path, images[plane]))
        if len({folder.resolve() for folder in folders}) != len(folders):
            raise ValueError(f"{plane} folders must not contain duplicates.")
        groups = tuple(map(list_images, folders))
        decoded = []
        for group in groups:
            values = []
            for path in group:
                image = read_labels(path)
                if (
                    num_phases is not None
                    and (largest := int(image.max())) >= num_phases
                ):
                    raise ValueError(
                        f"phase labels must be in [0, {num_phases - 1}], "
                        f"found {largest}: {path}"
                    )
                values.append(image)
            decoded.append(tuple(values))
        planes[plane] = PlaneImages(
            plane=plane,
            folders=folders,
            paths=groups,
            images=tuple(decoded),
        )
    return planes


def infer_num_phases(planes):
    present = np.zeros(256, dtype=bool)
    largest, source = -1, None
    for images in planes.values():
        for group, paths in zip(images.images, images.paths, strict=True):
            for image, path in zip(group, paths, strict=True):
                labels = np.flatnonzero(np.bincount(image.ravel()))
                present[labels] = True
                if labels[-1] > largest:
                    largest, source = int(labels[-1]), path
    count = largest + 1
    missing = np.flatnonzero(~present[:count])
    if missing.size:
        raise ValueError(
            f"label {missing[0]} never appears below the largest label {count - 1}; "
            "use labels 0..K-1, or set num_phases to keep an unused phase. "
            f"Largest label in: {source}"
        )
    return validate_num_phases(count)


def fingerprint(planes):
    hashes = {}
    result = {plane: {} for plane in planes}
    for plane, images in planes.items():
        for index, (folder, paths) in enumerate(
            zip(images.folders, images.paths, strict=True)
        ):
            prefix = "" if len(images.folders) == 1 else f"{index}/"
            for path in paths:
                if path not in hashes:
                    hashes[path] = file_hash(path)
                key = prefix + path.relative_to(folder).as_posix()
                result[plane][key] = hashes[path]
    return result
