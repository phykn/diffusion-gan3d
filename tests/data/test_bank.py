import pytest
import torch

from src.data.bank import validate_bank, validate_bank_height


def test_bank_validation_returns_fractional_volumes_without_copying():
    volumes = torch.full((1, 2, 2, 2, 2), 0.5)
    bank = {0: volumes}

    validated = validate_bank(bank, {0: {}}, size=2, phases=2)

    assert validated is bank
    assert validated[0] is volumes
    torch.testing.assert_close(volumes, torch.full_like(volumes, 0.5))


@pytest.mark.parametrize("invalid", ["domain", "shape", "fractions"])
def test_bank_validation_preserves_specific_errors(invalid):
    bank = {0: torch.full((1, 2, 2, 2, 2), 0.5)}
    if invalid == "domain":
        bank[1] = bank.pop(0)
        message = "domains must match"
    elif invalid == "shape":
        bank[0] = bank[0][..., :1]
        message = "must contain"
    else:
        bank[0].fill_(0.25)
        message = "invalid phase fractions"

    with pytest.raises(ValueError, match=message):
        validate_bank(bank, {0: {}}, size=2, phases=2)


def test_bank_height_keeps_per_sample_tensors_connected_to_refresh():
    bank = {0: torch.full((2, 2, 2, 2, 2), 0.5)}
    origins = {0: torch.tensor([0.0, 4.0])}
    extents = {0: torch.tensor([16.0, 24.0])}
    resolved = validate_bank_height(bank, origins, extents, crop_size=8)

    assert resolved[0] is extents[0]
    extents[0][1] = 32.0
    assert resolved[0][1] == 32.0
    torch.testing.assert_close(origins[0], torch.tensor([0.0, 4.0]))


@pytest.mark.parametrize(
    "origins,extents,message",
    [
        (None, {0: 16.0}, "requires bank crop origins"),
        ({1: torch.zeros(2)}, {0: 16.0}, "requires bank crop origins"),
        ({0: torch.zeros(2)}, {1: 16.0}, "extent domains must match"),
        ({0: torch.zeros(2)}, {0: None}, "requires per-sample height extents"),
        ({0: torch.zeros(2)}, {0: torch.ones(3)}, "invalid LR bank height extents"),
        ({0: torch.zeros(2)}, {0: float("nan")}, "invalid LR bank height extents"),
        ({0: torch.zeros(2)}, {0: 0.0}, "invalid LR bank height extents"),
        ({0: torch.zeros(1)}, {0: 16.0}, "invalid LR bank crop origins"),
        ({0: torch.tensor([-1.0, 0.0])}, {0: 16.0}, "invalid LR bank crop origins"),
        (
            {0: torch.tensor([0.0, float("inf")])},
            {0: 16.0},
            "invalid LR bank crop origins",
        ),
        (
            {0: torch.tensor([0.0, 9.0])},
            {0: torch.tensor([16.0, 16.0])},
            "invalid LR bank crop origins",
        ),
    ],
)
def test_bank_height_preserves_coordinate_validation(origins, extents, message):
    bank = {0: torch.full((2, 2, 2, 2, 2), 0.5)}
    with pytest.raises(ValueError, match=message):
        validate_bank_height(bank, origins, extents, crop_size=8)
