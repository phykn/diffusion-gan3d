import torch

from src.data.real import RealDataset
from src.prepare.resize import resize_crop


class ResolutionDataset(RealDataset):
    def __init__(self, path_groups, crop_size, patch_size, num_phases):
        super().__init__(path_groups, crop_size=crop_size, patch_size=patch_size)
        self.num_phases = num_phases

    def __getitem__(self, path) -> torch.Tensor:
        labels = torch.from_numpy(self.crop(self.decode(path)).copy()).long()
        return resize_crop(labels, self.patch_size, self.num_phases)
