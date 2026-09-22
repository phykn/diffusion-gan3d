from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.split import training_origin
from src.plane import get_axis
from src.prepare.resize import resize_crop


class RealDataset(Dataset[dict]):
    """Image and source geometry; z is the row direction of xz/yz images."""

    def __init__(
        self,
        path_groups: Sequence[Sequence[str | Path]],
        crop_size: int,
        patch_size: int,
        num_phases: int,
        plane: str | int = "xz",
        validation_regions: dict | None = None,
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
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.axis = get_axis(plane)
        self.height_direction = 0 if self.axis != 0 else None
        self.validation_regions = validation_regions or {}

    def __len__(self) -> int:
        return sum(len(group) for group in self.path_groups)

    def __getitem__(self, path: str | Path) -> dict:
        path = Path(path).resolve()
        source = self.decode(path)
        excluded = self.validation_regions.get(str(path))
        crop, origin = self.crop_with_origin(source, excluded)
        labels = torch.from_numpy(crop.copy()).long()
        image = resize_crop(labels, self.patch_size, self.num_phases)
        return {
            "image": image,
            "image_id": str(path),
            "source_shape": torch.tensor(source.shape),
            "crop_origin": torch.tensor(origin),
            "height_extent": float(source.shape[self.height_direction])
            if self.height_direction is not None
            else -1.0,
            "height_origin": float(origin[self.height_direction])
            if self.height_direction is not None
            else -1.0,
        }

    def decode(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            data = np.asarray(img)
        return self.check_image(np.array(data, copy=True))

    def crop_with_origin(
        self, img: np.ndarray, excluded: Sequence[int] | None = None
    ) -> tuple[np.ndarray, tuple[int, int]]:
        img = self.check_image(img)
        height, width = img.shape
        size = self.crop_size
        if size > min(height, width):
            raise ValueError("crop size must fit inside the image.")
        if excluded is None:
            top = int(np.random.randint(0, height - size + 1))
            left = int(np.random.randint(0, width - size + 1))
        else:
            top, left = training_origin(img.shape, size, excluded)
        return img[top : top + size, left : left + size], (top, left)

    @staticmethod
    def check_image(img: np.ndarray) -> np.ndarray:
        if img.ndim != 2:
            raise ValueError("image must be two-dimensional.")
        if img.dtype != np.uint8:
            raise ValueError("image must use uint8.")
        return img
