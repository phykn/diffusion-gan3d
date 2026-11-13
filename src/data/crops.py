from dataclasses import dataclass

import numpy as np
import torch

from src.data.phases import encode_phases, resize_phases


@dataclass(frozen=True)
class Crops:
    images: torch.Tensor
    origins: torch.Tensor
    extents: torch.Tensor


class CropSampler:
    def __init__(self, plane, crop_size, patch_size, num_phases):
        self.groups = plane.images
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_phases = num_phases
        for group, paths in zip(plane.images, plane.paths):
            for image, path in zip(group, paths):
                if crop_size > min(image.shape):
                    raise ValueError(
                        f"crop_size {crop_size} does not fit inside {path} "
                        f"({image.shape[0]}x{image.shape[1]})."
                    )

    def sample(self, count, generator):
        group = self.groups[
            int(torch.randint(len(self.groups), (), generator=generator))
        ]
        choices = torch.randint(len(group), (count,), generator=generator).tolist()
        size = self.crop_size
        crops, origins, extents = [], [], []
        for choice in choices:
            image = group[choice]
            height, width = image.shape
            top = int(torch.randint(height - size + 1, (), generator=generator))
            left = int(torch.randint(width - size + 1, (), generator=generator))
            crops.append(image[top : top + size, left : left + size])
            origins.append(float(top))
            extents.append(float(height))
        probs = encode_phases(torch.from_numpy(np.stack(crops)), self.num_phases)
        return Crops(
            images=resize_phases(probs, (self.patch_size,) * 2),
            origins=torch.tensor(origins),
            extents=torch.tensor(extents),
        )
