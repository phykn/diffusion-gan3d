import torch

from src.data.real import RealDataset
from src.prepare.resize import resize_crop


class ResolutionDataset(RealDataset):
    def __init__(
        self,
        path_groups,
        crop_size,
        patch_size,
        num_phases,
        height_direction=None,
        height_enabled=False,
    ):
        super().__init__(path_groups, crop_size=crop_size, patch_size=patch_size)
        self.num_phases = num_phases
        self.height_direction = height_direction
        self.height_enabled = height_enabled

    def __getitem__(self, path) -> torch.Tensor:
        crop, origin = self.crop_with_origin(self.decode(path))
        labels = torch.from_numpy(crop.copy()).long()
        image = resize_crop(labels, self.patch_size, self.num_phases)
        if not self.height_enabled:
            return image
        return {
            "image": image,
            "height_origin": float(origin[self.height_direction])
            if self.height_direction is not None
            else -1.0,
        }
