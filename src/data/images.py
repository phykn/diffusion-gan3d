from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.axes import PLANES

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".pgm"}


@dataclass(frozen=True)
class PlaneImages:
    paths: tuple
    images: tuple


def load_images(folders, num_phases):
    planes = {}
    loaded = {}
    for plane in PLANES:
        if plane not in folders:
            continue
        sources = folders[plane]
        if len({folder.resolve() for folder in sources}) != len(sources):
            raise ValueError(f"{plane} folders must not contain duplicates.")
        groups = []
        for folder in sources:
            key = folder.resolve()
            if key not in loaded:
                paths = list_images(folder)
                images = tuple(read_labels(path, num_phases) for path in paths)
                loaded[key] = paths, images
            groups.append(loaded[key])
        planes[plane] = PlaneImages(*zip(*groups))
    return planes


def list_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder does not exist: {folder}")
    paths = tuple(
        sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )
    if not paths:
        raise ValueError(f"image folder contains no images: {folder}")
    return paths


def read_labels(path, num_phases=None):
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"training images must contain one 2D frame: {path}")
            labels = np.asarray(image)
    except OSError as exc:
        raise ValueError(f"cannot read image {path}: {exc}") from exc
    if labels.ndim != 2:
        raise ValueError(f"training images must be two-dimensional: {path}")
    if labels.dtype != np.uint8:
        raise ValueError(f"training images must use uint8 labels: {path}")
    if num_phases is not None and (largest := int(labels.max())) >= num_phases:
        raise ValueError(
            f"phase labels must be in [0, {num_phases - 1}], found {largest}: {path}"
        )
    return labels


def infer_num_phases(planes):
    present = np.zeros(256, dtype=bool)
    largest, source = -1, None
    for plane in planes.values():
        for group, paths in zip(plane.images, plane.paths):
            for image, path in zip(group, paths):
                labels = np.flatnonzero(np.bincount(image.ravel()))
                present[labels] = True
                if labels[-1] > largest:
                    largest, source = int(labels[-1]), path
    missing = np.flatnonzero(~present[:largest])
    if missing.size:
        raise ValueError(
            f"label {missing[0]} never appears below the largest label {largest}; "
            "use labels 0..K-1, or set num_phases to keep an unused phase. "
            f"Largest label in: {source}"
        )
    return largest + 1


def list_side_images(planes):
    return [
        image
        for name, plane in planes.items()
        if name != PLANES[0]
        for group in plane.images
        for image in group
    ]
