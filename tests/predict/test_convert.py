from types import SimpleNamespace

import pytest
import torch

from src.predict import convert
from src.predict.generator import Generator
from src.predict.tiling.state import write_output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_owned_probability_conversion_matches_formula_and_reuses_float32(dtype):
    clean = torch.linspace(-1.1, 1.1, 120, dtype=dtype).reshape(1, 3, 2, 4, 5)
    original = clean.clone()
    expected = ((clean.float() + 1) * 0.5).clamp(0, 1)
    expected = expected / expected.sum(1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    )
    result = convert.owned_clean_to_probs_(clean)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == torch.float32
    if dtype == torch.float32:
        assert result.data_ptr() == clean.data_ptr()
    else:
        torch.testing.assert_close(clean, original)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_labels_use_bounded_argmax_with_identical_ties_and_strided_input(
    monkeypatch, device
):
    channels = torch.randn(3, 11, 6, 9, device=device).transpose(2, 3)
    channels[:, 0] = 0  # Preserve first-channel tie breaking.
    expected = channels.argmax(0).to(device="cpu", dtype=torch.uint8)
    monkeypatch.setattr(convert, "LABEL_CHUNK_VOXELS", 2 * 6 * 9)
    original = torch.Tensor.argmax
    shapes = []

    def bounded(tensor, *args, **kwargs):
        shapes.append(tensor.shape)
        assert tensor.shape[1] <= 2
        assert tensor.device.type == device
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "argmax", bounded)
    actual = convert.labels_from_channels(channels)
    assert len(shapes) == 6
    assert actual.device.type == "cpu" and actual.dtype == torch.uint8
    torch.testing.assert_close(actual, expected)


def test_uint8_labels_preserve_phase_255_and_reject_phase_256():
    channels = torch.arange(256.0).reshape(256, 1, 1, 1)
    assert convert.labels_from_channels(channels).item() == 255
    labels = torch.zeros(1, 1, 1, dtype=torch.uint8)
    region = (slice(None),) * 3
    write_output(labels, region, channels.unsqueeze(0))
    assert labels.item() == 255
    channels = torch.arange(257.0).reshape(257, 1, 1, 1)
    with pytest.raises(ValueError, match="num_phases"):
        convert.labels_from_channels(channels)
    with pytest.raises(ValueError, match="num_phases"):
        write_output(labels, region, channels.unsqueeze(0))
    assert labels.item() == 255


@pytest.mark.parametrize("shape", [(2, 2, 2), (0, 1, 1, 1), (2, 0, 1, 1)])
def test_label_conversion_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="C,D,H,W"):
        convert.labels_from_channels(torch.empty(shape))


@pytest.mark.parametrize("phases", [0, 257, True, 2.5])
def test_generator_rejects_unsupported_phases_before_sampling(phases):
    model = SimpleNamespace(num_domains=1, downsample_factor=1)
    with pytest.raises(ValueError, match="num_phases"):
        Generator(model, None, torch.device("cpu"), 4, phases, 4, False)
