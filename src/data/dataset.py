from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .image import load_labels
from .patch import random_crop, resize_labels


class LabelPatchDataset(Dataset[torch.Tensor]):
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
        labels = load_labels(self.paths[index])
        if int(labels.max()) >= self.num_phases:
            raise ValueError(
                f"label image must contain phases 0 to {self.num_phases - 1}."
            )
        labels = random_crop(labels, self.crop_size)
        labels = resize_labels(labels, self.patch_size)
        return torch.from_numpy(labels.copy()).to(torch.long)
