from dataclasses import dataclass

import numpy as np
import torch

from src.data.phases import encode_phases
from src.data.resize import resize_phases


@dataclass(frozen=True)
class Crops:
    images: torch.Tensor
    height_origins: torch.Tensor
    height_extents: torch.Tensor


class CropSampler:
    def __init__(self, plane, crop_size, patch_size, num_phases):
        self.folders = plane.images
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_phases = num_phases
        for images, paths in zip(plane.images, plane.paths):
            for image, path in zip(images, paths):
                if crop_size > min(image.shape):
                    raise ValueError(
                        f"crop_size {crop_size} does not fit inside {path} "
                        f"({image.shape[0]}x{image.shape[1]})."
                    )

    def sample(self, count, generator):
        images = self.folders[self._draw(len(self.folders), generator)]
        picks = torch.randint(len(images), (count,), generator=generator).tolist()
        size = self.crop_size
        crops, top_rows, image_heights = [], [], []
        for pick in picks:
            image = images[pick]
            height, width = image.shape
            top = self._draw(height - size + 1, generator)
            left = self._draw(width - size + 1, generator)
            crops.append(image[top : top + size, left : left + size])
            top_rows.append(float(top))
            image_heights.append(float(height))
        labels = torch.from_numpy(np.stack(crops))
        fractions = encode_phases(labels, self.num_phases)
        return Crops(
            images=resize_phases(fractions, (self.patch_size,) * 2),
            height_origins=torch.tensor(top_rows),
            height_extents=torch.tensor(image_heights),
        )

    @staticmethod
    def _draw(limit, generator):
        return int(torch.randint(limit, (), generator=generator))
