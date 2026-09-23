from unittest.mock import patch

import pytest
import torch

from src.data.augment import augment_volumes


@pytest.mark.parametrize("height,count", [(False, 48), (True, 8)])
def test_volume_symmetries_preserve_fractions_and_height(height, count):
    values = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3) / 27
    source = torch.cat((values, 1 - values), dim=1).expand(count, -1, -1, -1, -1)
    before = source.clone()
    with patch("src.data.augment.torch.randint", return_value=torch.arange(count)):
        actual = augment_volumes(source, preserve_height=height)
    assert torch.unique(actual.flatten(1), dim=0).shape[0] == count
    assert torch.equal(source, before)
    assert actual.dtype == source.dtype and actual.device == source.device
    torch.testing.assert_close(actual.sum(1), torch.ones_like(actual[:, 0]))
    torch.testing.assert_close(
        actual.flatten(2).sort().values, source.flatten(2).sort().values
    )
    if height:
        torch.testing.assert_close(
            actual.flatten(3).sort().values, source.flatten(3).sort().values
        )
    else:
        assert any(
            not torch.equal(row.mean((-1, -2)), source[0].mean((-1, -2)))
            for row in actual
        )


def test_volume_augmentation_is_seeded_and_preserves_rectangular_shape():
    source = torch.arange(2 * 3 * 4 * 5).reshape(2, 1, 3, 4, 5)
    torch.manual_seed(71)
    first = augment_volumes(source)
    torch.manual_seed(71)
    second = augment_volumes(source)
    assert torch.equal(first, second)
    assert first.shape == source.shape
