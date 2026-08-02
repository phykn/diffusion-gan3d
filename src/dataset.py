from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SliceDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        paths: Sequence[str | Path],
        crop_size: int = 64,
        patch_size: int = 64,
        augment: bool = False,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("paths must not be empty.")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in (crop_size, patch_size)
        ):
            raise ValueError("crop and patch sizes must be positive integers.")
        self.crop_size = crop_size
        self.patch_size = patch_size
        if not isinstance(augment, bool):
            raise TypeError("augment must be a boolean.")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.load(self.paths[idx])
        img = self.crop(img)
        img = self.resize(img)
        if self.augment:
            img = self.transform(img, int(np.random.randint(8)))
        return torch.from_numpy(img.copy()).to(torch.long)

    def load(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            data = np.asarray(img)
        self.check_image(data)
        return np.array(data, copy=True)

    def crop(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)
        h, w = img.shape
        if self.crop_size > min(h, w):
            raise ValueError("crop size must fit inside the image.")
        top = int(np.random.randint(0, h - self.crop_size + 1))
        left = int(np.random.randint(0, w - self.crop_size + 1))
        return img[top : top + self.crop_size, left : left + self.crop_size]

    def resize(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)
        if img.shape == (self.patch_size, self.patch_size):
            return img
        return np.asarray(
            Image.fromarray(img).resize(
                (self.patch_size, self.patch_size),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )

    @staticmethod
    def transform(img: np.ndarray, idx: int) -> np.ndarray:
        if idx >= 4:
            img = np.flip(img, axis=1)
        return np.rot90(img, idx % 4)

    @staticmethod
    def check_image(img: np.ndarray) -> None:
        if img.ndim != 2:
            raise ValueError("image must be two-dimensional.")
        if img.dtype != np.uint8:
            raise ValueError("image must use uint8.")


class BatchStream:
    def __init__(self, loader: DataLoader[torch.Tensor]) -> None:
        self.loader = loader
        self.iterator: Iterator[torch.Tensor] = iter(loader)

    def next(self) -> torch.Tensor:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
