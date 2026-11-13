import torch

from src.config.schema import update
from src.data.augment import augment_volumes, build_volume_transforms


def test_bank_without_flips_is_identity(tiny_configs):
    config = update(tiny_configs["sr"], "lr_bank.flips", False)
    transforms = build_volume_transforms(config)
    volumes = torch.rand(3, 2, 4, 4, 4)
    for seed in range(100):
        generator = torch.Generator().manual_seed(seed)
        assert torch.equal(augment_volumes(volumes, transforms, generator), volumes)


def test_bank_with_flips_uses_cube_symmetries(tiny_configs):
    transforms = build_volume_transforms(tiny_configs["sr"])
    assert len(transforms.axis_orders) * 2 ** len(transforms.flippable_axes) == 48
