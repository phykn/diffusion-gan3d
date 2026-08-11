from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


class SliceDataset(Dataset[torch.Tensor]):
    CACHE_BYTES = 64 * 1024**2

    def __init__(
        self,
        paths: Sequence[str | Path],
        crop_size: int = 64,
        patch_size: int = 64,
        allow_partial_crop: bool = False,
        batch_groups: Sequence[Sequence[int]] | None = None,
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
        groups = (
            (tuple(range(len(self.paths))),)
            if batch_groups is None
            else tuple(tuple(group) for group in batch_groups)
        )
        flattened = tuple(index for group in groups for index in group)
        if (
            not groups
            or any(not group for group in groups)
            or sorted(flattened) != list(range(len(self.paths)))
        ):
            raise ValueError("batch groups must partition the dataset indices.")
        self.batch_groups = groups
        self._cache_bytes = 0

    @classmethod
    def from_path_groups(
        cls,
        path_groups: Sequence[Sequence[str | Path]],
        *,
        crop_size: int = 64,
        patch_size: int = 64,
        allow_partial_crop: bool = False,
    ) -> "SliceDataset":
        groups = tuple(tuple(Path(path) for path in group) for group in path_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("path groups must be non-empty.")
        paths = tuple(path for group in groups for path in group)
        batch_groups = []
        offset = 0
        for group in groups:
            end = offset + len(group)
            batch_groups.append(tuple(range(offset, end)))
            offset = end
        return cls(
            paths,
            crop_size=crop_size,
            patch_size=patch_size,
            allow_partial_crop=allow_partial_crop,
            batch_groups=batch_groups,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.load(self.paths[idx])
        img = self.crop(img)
        img = self.resize(img)
        return torch.from_numpy(img.copy()).to(torch.long)

    def output_shape(self, idx: int) -> tuple[int, int]:
        with Image.open(self.paths[idx]) as image:
            width, height = image.size
        if not self.allow_partial_crop:
            if self.crop_size > min(height, width):
                raise ValueError(
                    f"crop size must fit inside the image: {self.paths[idx]}"
                )
            return (self.patch_size, self.patch_size)
        crop_h = min(height, self.crop_size)
        crop_w = min(width, self.crop_size)
        return self._resize_shape(crop_h, crop_w)

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
        output_h, output_w = self._resize_shape(*img.shape)
        if img.shape == (output_h, output_w):
            return img
        return np.asarray(
            Image.fromarray(img).resize(
                (output_w, output_h),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )

    def _resize_shape(self, height: int, width: int) -> tuple[int, int]:
        if not self.allow_partial_crop:
            return (self.patch_size, self.patch_size)
        return (
            max(1, round(height * self.patch_size / self.crop_size)),
            max(1, round(width * self.patch_size / self.crop_size)),
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


def folder_weights(counts: Sequence[int]) -> torch.Tensor:
    values = tuple(counts)
    if not values or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 1
        for count in values
    ):
        raise ValueError("folder counts must be positive integers.")
    return torch.log1p(torch.tensor(values, dtype=torch.float64))


class FolderBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: SliceDataset, batch_size: int) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch size must be a positive integer.")
        self.batch_size = batch_size
        self.num_batches = max(len(dataset), batch_size) // batch_size
        buckets = []
        for group in dataset.batch_groups:
            shapes = {dataset.output_shape(index) for index in group}
            if len(shapes) != 1:
                paths = ", ".join(str(dataset.paths[index]) for index in group)
                raise ValueError(
                    f"images in one axis folder must produce the same shape: {paths}"
                )
            buckets.append(group)
        self._buckets = tuple(buckets)
        self._weights = folder_weights(len(bucket) for bucket in self._buckets)

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            bucket_index = int(
                torch.multinomial(self._weights, 1, replacement=True).item()
            )
            bucket = self._buckets[bucket_index]
            choices = torch.randint(len(bucket), (self.batch_size,))
            yield [bucket[int(choice)] for choice in choices]

    def __len__(self) -> int:
        return self.num_batches
