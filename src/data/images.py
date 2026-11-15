from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.axes import PLANES


@dataclass(frozen=True)
class PlaneImages:
    paths: tuple
    images: tuple


def load_images(folders_by_plane, num_phases):
    planes = {}
    already_read = {}
    for plane in PLANES:
        if plane not in folders_by_plane:
            continue
        folders = folders_by_plane[plane]
        if len({folder.resolve() for folder in folders}) != len(folders):
            raise ValueError(f"{plane} folders must not contain duplicates.")
        paths_per_folder, images_per_folder = [], []
        for folder in folders:
            key = folder.resolve()
            if key not in already_read:
                paths = list_images(folder)
                images = tuple(read_labels(path, num_phases) for path in paths)
                already_read[key] = paths, images
            paths, images = already_read[key]
            paths_per_folder.append(paths)
            images_per_folder.append(images)
        planes[plane] = PlaneImages(tuple(paths_per_folder), tuple(images_per_folder))
    return planes


def list_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder does not exist: {folder}")
    paths = tuple(
        sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".pgm"}
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
    except OSError as error:
        raise ValueError(f"cannot read image {path}: {error}") from error
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
    label_seen = np.zeros(256, dtype=bool)
    largest, largest_path = -1, None
    for plane in planes.values():
        for images, paths in zip(plane.images, plane.paths):
            for image, path in zip(images, paths):
                labels = np.unique(image)
                label_seen[labels] = True
                if labels[-1] > largest:
                    largest, largest_path = int(labels[-1]), path
    missing = np.flatnonzero(~label_seen[:largest])
    if missing.size:
        raise ValueError(
            f"label {missing[0]} never appears below the largest label {largest}; "
            "use labels 0..K-1, or set num_phases to keep an unused phase. "
            f"Largest label in: {largest_path}"
        )
    return largest + 1


def list_side_images(planes):
    return [
        image
        for name, plane in planes.items()
        if name != PLANES[0]
        for images in plane.images
        for image in images
    ]
