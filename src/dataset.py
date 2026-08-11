from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SliceDataset(Dataset[torch.Tensor]):
    CACHE_BYTES = 64 * 1024**2

    def __init__(
        self,
        paths: Sequence[str | Path],
        crop_size: int = 64,
        patch_size: int = 64,
        allow_partial_crop: bool = False,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("paths must not be empty.")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in (crop_size, patch_size)
        ):
            raise ValueError("crop and patch sizes must be positive integers.")
        if not isinstance(allow_partial_crop, bool):
            raise TypeError("allow_partial_crop must be a boolean.")
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.allow_partial_crop = allow_partial_crop
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._cache_bytes = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.load(self.paths[idx])
        img = self.crop(img)
        img = self.resize(img)
        return torch.from_numpy(img.copy()).to(torch.long)

    def load(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        data = self.decode(path)
        if data.nbytes <= self.CACHE_BYTES:
            while self._cache and self._cache_bytes + data.nbytes > self.CACHE_BYTES:
                _, removed = self._cache.popitem(last=False)
                self._cache_bytes -= removed.nbytes
            self._cache[path] = data
            self._cache_bytes += data.nbytes
        return data

    def decode(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            data = np.asarray(img)
        self.check_image(data)
        return np.array(data, copy=True)

    def crop(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)
        h, w = img.shape
        if not self.allow_partial_crop and self.crop_size > min(h, w):
            raise ValueError("crop size must fit inside the image.")
        crop_h = min(h, self.crop_size) if self.allow_partial_crop else self.crop_size
        crop_w = min(w, self.crop_size) if self.allow_partial_crop else self.crop_size
        top = int(np.random.randint(0, h - crop_h + 1))
        left = int(np.random.randint(0, w - crop_w + 1))
        return img[top : top + crop_h, left : left + crop_w]

    def resize(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)
        if self.allow_partial_crop:
            h, w = img.shape
            output_h = max(1, round(h * self.patch_size / self.crop_size))
            output_w = max(1, round(w * self.patch_size / self.crop_size))
        else:
            output_h = output_w = self.patch_size
        if img.shape == (output_h, output_w):
            return img
        return np.asarray(
            Image.fromarray(img).resize(
                (output_w, output_h),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )

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
