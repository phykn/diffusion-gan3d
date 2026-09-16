from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

from src.config import get_domains, get_sizes, get_sr_sizes, normalize_train_config
from src.data.augment import CriticAugment
from src.data.loader import BatchStream, FolderBatchSampler
from src.data.real import RealDataset
from src.data.resolution import ResolutionDataset
from src.plane import PLANE_DIRECTIONS, PLANES

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def resolve_height_metadata(cfg):
    if not cfg["conditioning"]["height_enabled"]:
        return
    data = cfg["data"]
    thickness = data.get("thickness_axis")
    if thickness not in ("x", "y", "z"):
        raise ValueError("height conditioning requires data.thickness_axis.")
    extents = {}
    for domain, folders in get_domains(data).items():
        sizes = set()
        for axis, paths in folders.items():
            directions = PLANE_DIRECTIONS[PLANES[axis]]
            if thickness not in directions:
                continue
            direction = directions.index(thickness)
            for folder in paths:
                for path in Path(folder).iterdir():
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        with Image.open(path) as image:
                            sizes.add(image.size[1 - direction])
        if len(sizes) != 1:
            raise ValueError(
                "height conditioning requires full-thickness side images with one consistent extent per domain."
            )
        extents[domain] = sizes.pop()
    if data.get("height_extents", extents) != extents:
        raise ValueError("side-image thickness differs from saved height_extents.")
    data["height_extents"] = extents


def build_augmentation(cfg: dict) -> CriticAugment:
    settings = cfg.get("augmentation", {})
    planes = settings.get("planes")
    augment = CriticAugment(
        prob=settings.get("probability", 0.5),
        planes=planes,
        thickness_axis=cfg["data"].get("thickness_axis"),
    )
    active = {axis for axes in get_domains(cfg["data"]).values() for axis in axes}
    if planes is not None and any(PLANES[axis] not in planes for axis in active):
        raise ValueError("augmentation.planes must define every observed plane.")
    return augment


def build_datasets(cfg: dict, high: bool = False) -> dict[int, dict[int, RealDataset]]:
    cfg = normalize_train_config(cfg, cfg.get("stage", "sr" if high else "low_res"))
    data = cfg["data"]
    if high:
        crop, low, high_size = get_sr_sizes(cfg)
    else:
        crop, low, high_size = get_sizes(data)
    datasets = {}
    for domain_id, folders in get_domains(data).items():
        grouped = {}
        for axis in sorted(folders):
            values = folders[axis]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"axis {axis} folders must be a sequence of paths.")
            if not values:
                raise ValueError(f"axis {axis} folders must not be empty.")
            axis_folders = tuple(Path(value) for value in values)
            if len({folder.resolve() for folder in axis_folders}) != len(axis_folders):
                raise ValueError(f"axis {axis} folders must not contain duplicates.")

            groups = []
            for folder in axis_folders:
                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"axis {axis} folder does not exist: {folder}"
                    )
                found = sorted(
                    path
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not found:
                    raise ValueError(f"axis {axis} folder contains no images: {folder}")
                groups.append(tuple(found))
            grouped[axis] = tuple(groups)

        datasets[domain_id] = {
            axis: ResolutionDataset(
                path_groups,
                crop,
                high_size if high else low,
                data["num_phases"],
                height_direction=(
                    PLANE_DIRECTIONS[PLANES[axis]].index(data["thickness_axis"])
                    if data.get("thickness_axis") in PLANE_DIRECTIONS[PLANES[axis]]
                    else None
                ),
                height_enabled=cfg["conditioning"]["height_enabled"],
            )
            for axis, path_groups in grouped.items()
        }
    return datasets


def build_stream(
    dataset: RealDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    loader = DataLoader(
        dataset,
        batch_sampler=FolderBatchSampler(dataset, batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return BatchStream(loader)
