import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

# Bound the unfolded input held by one weight-gradient chunk; larger chunks
# barely help on MPS and raise peak memory.
CHUNK_BYTES = 64 * 2**20
# Smaller grids already take the native MPS weight gradient quickly.
MIN_VOXELS = 32**3


def conv3d_weight(x: torch.Tensor, grad: torch.Tensor, kernel: int) -> torch.Tensor:
    """Weight gradient of a stride-1, same-padded cube convolution via im2col."""
    batch, channels, depth, height, width = x.shape
    out_channels = grad.shape[1]
    pad = kernel // 2
    taps = kernel * kernel
    rows = grad.permute(0, 2, 3, 4, 1)
    slice_bytes = batch * channels * taps * height * width * x.element_size()
    step = max(1, CHUNK_BYTES // slice_bytes - 2 * pad)
    weight = x.new_zeros(kernel, channels * taps, out_channels)
    for start in range(0, depth, step):
        size = min(step, depth - start)
        # Unfold each depth slice in-plane once; depth taps then share its columns.
        low, high = max(start - pad, 0), min(start + size + pad, depth)
        window = F.pad(
            x[:, :, low:high],
            (0, 0, 0, 0, low - (start - pad), start + size + pad - high),
        ).transpose(1, 2)
        cols = F.unfold(
            window.reshape(-1, channels, height, width), kernel, padding=pad
        ).view(batch, size + 2 * pad, channels * taps, height * width)
        chunk = rows[:, start : start + size].reshape(
            batch * size, height * width, out_channels
        )
        for tap in range(kernel):
            part = cols[:, tap : tap + size].reshape(
                batch * size, channels * taps, height * width
            )
            weight[tap] += torch.bmm(part, chunk).sum(0)
    return weight.view(kernel, channels, kernel, kernel, out_channels).permute(
        4, 1, 0, 2, 3
    )


class _Conv3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return F.conv3d(x, weight, bias, padding=weight.shape[-1] // 2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        kernel = weight.shape[-1]
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.nn.grad.conv3d_input(
                x.shape, weight, grad, padding=kernel // 2
            )
        if ctx.needs_input_grad[1]:
            grad_weight = conv3d_weight(x, grad, kernel)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad.sum((0, 2, 3, 4))
        return grad_x, grad_weight, grad_bias


class Conv3d(nn.Conv3d):
    """nn.Conv3d whose MPS weight gradient avoids the slow native kernel."""

    def _conv_forward(self, x, weight, bias):
        if (
            x.device.type == "mps"
            and torch.is_grad_enabled()
            and weight.requires_grad
            and not torch.is_autocast_enabled("mps")
            and self.kernel_size == (3, 3, 3)
            and self.stride == (1, 1, 1)
            and self.padding == (1, 1, 1)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
            and self.padding_mode == "zeros"
            and x.shape[2:].numel() >= MIN_VOXELS
        ):
            return _Conv3dFunction.apply(x, weight, bias)
        return super()._conv_forward(x, weight, bias)
