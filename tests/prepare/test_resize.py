from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.prepare.resize import downsample, phase_channels, resize_phases


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
        torch.ones(1, 2, 2, dtype=torch.complex64) * 1j,
    ],
)
def test_phase_channel_validation_is_preserved(labels):
    with pytest.raises(ValueError):
        phase_channels(labels, 3)


@pytest.mark.parametrize(
    "dtype,count", [(torch.uint8, 256), (torch.int8, 128), (torch.int16, 32768)]
)
def test_phase_channels_validates_without_small_integer_overflow(dtype, count):
    labels = torch.tensor([[[0, count - 1]]], dtype=dtype)
    result = phase_channels(labels, count)
    torch.testing.assert_close(result.argmax(1), labels.long())
    assert torch.equal(result.sum(1), torch.ones_like(labels, dtype=torch.float32))
    with pytest.raises(ValueError):
        phase_channels(labels, count - 1)


def test_mixed_resize_preserves_area_along_shrinking_axis():
    probs = torch.zeros(1, 2, 6, 2)
    probs[:, 0, (0, 5)] = 1
    probs[:, 1] = 1 - probs[:, 0]
    result = resize_phases(probs, (2, 4))
    expected = torch.tensor([1 / 3, 2 / 3]).reshape(1, 2, 1, 1).expand(1, 2, 2, 4)
    torch.testing.assert_close(result, expected)


def test_downsample_rejects_expansion():
    with pytest.raises(ValueError, match="expand"):
        downsample(torch.ones(1, 1, 2, 4), (4, 2))


def test_nonintegral_resize_integrates_pixel_area():
    labels = torch.zeros(1, 3, 3, dtype=torch.long)
    labels[0, 1, 1] = 1
    resized = resize_phases(phase_channels(labels, 2), (2, 2))
    torch.testing.assert_close(resized[0, 1], torch.full((2, 2), 1 / 9))


@pytest.mark.parametrize(
    "source,target",
    [
        ((7, 5), (3, 2)),
        ((3, 3), (5, 7)),
        ((7, 3), (3, 5)),
        ((7, 5, 9), (3, 2, 4)),
        ((3, 4, 5), (5, 8, 7)),
        ((8, 6, 4), (4, 3, 2)),
        ((3, 5), (6, 15)),
    ],
)
def test_resize_preserves_simplex_phase_amount_and_gradient(source, target):
    logits = torch.randn(2, 3, *source, requires_grad=True)
    probs = logits.softmax(1)
    actual = resize_phases(probs, target)
    dims = tuple(range(2, actual.ndim))
    torch.testing.assert_close(actual.mean(dims), probs.mean(dims))
    torch.testing.assert_close(actual.sum(1), torch.ones(2, *target))
    assert actual.min() >= 0
    actual.square().mean().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
