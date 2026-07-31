from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler

AXES = (0, 1, 2)
_EXTENSIONS = {".png", ".tif", ".tiff"}


class SliceDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        crop_size: int,
        patch_size: int,
        num_phases: int,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("paths must not be empty.")
        if crop_size <= 0 or patch_size <= 0:
            raise ValueError("crop and patch sizes must be positive.")
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")
        self.crop_size = int(crop_size)
        self.patch_size = int(patch_size)
        self.num_phases = int(num_phases)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        labels = _load(self.paths[index])
        if int(labels.max()) >= self.num_phases:
            raise ValueError(
                f"label image must contain phases 0 to {self.num_phases - 1}."
            )
        labels = crop_labels(labels, self.crop_size)
        labels = resize_labels(labels, self.patch_size)
        return torch.from_numpy(labels.copy()).to(torch.long)


class BatchStream:
    def __init__(self, loader: DataLoader[torch.Tensor]) -> None:
        self.loader = loader
        self._iterator: Iterator[torch.Tensor] = iter(loader)

    def next(self) -> torch.Tensor:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


def find_slices(folders: dict[int, Path]) -> dict[int, tuple[Path, ...]]:
    if set(folders) != set(AXES):
        raise ValueError("axis folders must contain exactly axes 0, 1, and 2.")

    grouped = {}
    for axis in AXES:
        folder = Path(folders[axis])
        if not folder.is_dir():
            raise FileNotFoundError(f"axis {axis} folder does not exist: {folder}")
        paths = tuple(
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in _EXTENSIONS
            )
        )
        if not paths:
            raise ValueError(f"axis {axis} folder contains no label images: {folder}")
        grouped[axis] = paths
    return grouped


def build_stream(
    dataset: SliceDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=max(1024, batch_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return BatchStream(loader)


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        labels = np.asarray(image)
    if labels.ndim != 2:
        raise ValueError(f"categorical image must be two-dimensional: {path}")
    if labels.dtype != np.uint8:
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"categorical image must contain integer labels: {path}")
        if labels.min() < 0 or labels.max() > 255:
            raise ValueError(f"categorical image labels must fit uint8: {path}")
        labels = labels.astype(np.uint8)
    return np.array(labels, copy=True)


def crop_labels(labels: np.ndarray, size: int) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional.")
    height, width = labels.shape
    if size <= 0 or size > min(height, width):
        raise ValueError("crop size must fit inside the image.")
    top = int(np.random.randint(0, height - size + 1))
    left = int(np.random.randint(0, width - size + 1))
    return labels[top : top + size, left : left + size]


def resize_labels(labels: np.ndarray, size: int) -> np.ndarray:
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("labels must be two-dimensional uint8 values.")
    if size <= 0:
        raise ValueError("resize size must be positive.")
    image = Image.fromarray(labels, mode="L")
    return np.asarray(
        image.resize((size, size), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
