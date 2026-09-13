from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RealDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        path_groups: Sequence[Sequence[str | Path]],
        crop_size: int = 64,
        patch_size: int = 64,
        allow_part: bool = False,
    ) -> None:
        self.path_groups = tuple(
            tuple(Path(path) for path in group) for group in path_groups
        )
        if not self.path_groups or any(not group for group in self.path_groups):
            raise ValueError("path groups must be non-empty.")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in (crop_size, patch_size)
        ):
            raise ValueError("crop and patch sizes must be positive integers.")
        if not isinstance(allow_part, bool):
            raise TypeError("allow_part must be a boolean.")

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.allow_part = allow_part

    def __len__(self) -> int:
        return sum(len(group) for group in self.path_groups)

    def __getitem__(self, path: str | Path) -> torch.Tensor:
        img = self.decode(Path(path))
        img = self.crop(img)
        img = self.resize(img)
        return torch.from_numpy(img.copy()).to(torch.long)

    def decode(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            data = np.asarray(img)

        self.check_image(data)
        return np.array(data, copy=True)

    def crop(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)

        h, w = img.shape
        if not self.allow_part and self.crop_size > min(h, w):
            raise ValueError("crop size must fit inside the image.")

        crop_h = min(h, self.crop_size) if self.allow_part else self.crop_size
        crop_w = min(w, self.crop_size) if self.allow_part else self.crop_size
        top = int(np.random.randint(0, h - crop_h + 1))
        left = int(np.random.randint(0, w - crop_w + 1))
        return img[top : top + crop_h, left : left + crop_w]

    def resize(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)
        height, width = img.shape

        if not self.allow_part:
            output_h = output_w = self.patch_size
        else:
            output_h = max(1, round(height * self.patch_size / self.crop_size))
            output_w = max(1, round(width * self.patch_size / self.crop_size))

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
