import torch

from src.config.schema import update
from src.data.augment import augment_volumes, build_symmetries


def test_bank_without_flips_is_identity(tiny_configs):
    config = update(tiny_configs["sr"], "lr_bank.flips", False)
    symmetries = build_symmetries(config)
    volumes = torch.rand(3, 2, 4, 4, 4)
    for seed in range(100):
        generator = torch.Generator().manual_seed(seed)
        assert torch.equal(augment_volumes(volumes, symmetries, generator), volumes)


def test_bank_with_flips_uses_cube_symmetries(tiny_configs):
    symmetries = build_symmetries(tiny_configs["sr"])
    assert len(symmetries.orders) * 2 ** len(symmetries.flip_axes) == 48
