import pytest
import torch

from src.data.bank import validate_bank


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
