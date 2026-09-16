from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.prepare.resize import phase_channels


@pytest.mark.parametrize("shape", [(2, 7, 5), (2, 6, 7, 5)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int64])
def test_phase_channels_avoids_expanded_int64_one_hot(shape, dtype):
    labels = torch.randint(0, 3, shape).to(dtype).transpose(-1, -2)
    expected = F.one_hot(labels.long(), 3).movedim(-1, 1).float()
    with patch.object(
        F, "one_hot", side_effect=AssertionError("expanded int64 allocation")
    ):
        actual = phase_channels(labels, 3)
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([[[-1]]]),
        torch.tensor([[[3]]]),
        torch.zeros(1, 2, 2, dtype=torch.float32),
    ],
)
def test_phase_channel_validation_is_preserved(labels):
    with pytest.raises(ValueError):
        phase_channels(labels, 3)
