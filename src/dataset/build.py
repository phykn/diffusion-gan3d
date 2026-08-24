from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import DataLoader

from ..config import get_domains
from .loader import BatchStream, FolderBatchSampler
from .real import RealDataset

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def build_datasets(cfg: dict) -> dict[int, dict[int, RealDataset]]:
    data = cfg["data"]
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
            axis: RealDataset(
                path_groups,
                crop_size=data["crop_size"],
                patch_size=data["input_size"],
                allow_part=data["allow_part"],
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
