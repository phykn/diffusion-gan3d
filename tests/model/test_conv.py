import pytest
import torch
from torch import nn

from src.model import conv
from src.model.conv import Conv3d, conv3d_weight

MPS = torch.backends.mps.is_available()


@pytest.mark.parametrize("chunk_bytes", [conv.CHUNK_BYTES, 1])
def test_unfolded_weight_gradient_matches_the_native_one(monkeypatch, chunk_bytes):
    # One-byte chunks split every depth slice, covering both padded ends.
    monkeypatch.setattr(conv, "CHUNK_BYTES", chunk_bytes)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 6, 7, dtype=torch.float64)
    grad = torch.randn(2, 4, 5, 6, 7, dtype=torch.float64)
    expected = torch.nn.grad.conv3d_weight(x, (4, 3, 3, 3, 3), grad, padding=1)
    torch.testing.assert_close(conv3d_weight(x, grad, 3), expected)


def test_cpu_keeps_the_native_convolution():
    layer = Conv3d(2, 3, 3, padding=1)
    out = layer(torch.randn(1, 2, 32, 32, 32))
    assert type(out.grad_fn).__name__ == "ConvolutionBackward0"
    assert set(layer.state_dict()) == set(nn.Conv3d(2, 3, 3).state_dict())


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
@pytest.mark.parametrize("bias", [True, False])
def test_apple_gpu_gradients_match_the_native_convolution(bias):
    torch.manual_seed(0)
    native = nn.Conv3d(3, 4, 3, padding=1, bias=bias).to("mps")
    layer = Conv3d(3, 4, 3, padding=1, bias=bias).to("mps")
    layer.load_state_dict(native.state_dict())
    x = torch.randn(2, 3, 32, 32, 32, device="mps")
    grads = []
    for module in (native, layer):
        inputs = x.clone().requires_grad_()
        out = module(inputs)
        out.square().mean().backward()
        grads.append([inputs.grad, *(p.grad for p in module.parameters())])
    assert type(out.grad_fn).__name__ == "_Conv3dFunctionBackward"
    for fast, reference in zip(*grads):
        torch.testing.assert_close(fast, reference, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_apple_gpu_falls_back_outside_the_fast_case():
    x = torch.randn(1, 2, 32, 32, 32, device="mps")
    strided = Conv3d(2, 3, 3, stride=2, padding=1).to("mps")
    assert type(strided(x).grad_fn).__name__ == "ConvolutionBackward0"
    small = Conv3d(2, 3, 3, padding=1).to("mps")
    assert type(small(x[..., :8, :8, :8]).grad_fn).__name__ == "ConvolutionBackward0"
    with torch.no_grad():
        torch.testing.assert_close(
            small(x), nn.functional.conv3d(x, small.weight, small.bias, padding=1)
        )
