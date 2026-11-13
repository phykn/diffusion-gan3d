import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

# Larger weight-gradient chunks barely help on MPS and raise peak memory.
CHUNK_BYTES = 64 * 2**20
# Smaller grids already take the native MPS weight gradient quickly.
MIN_VOXELS = 32**3


class _Conv3dFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="mps", cast_inputs=torch.float16)
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return F.conv3d(x, weight, bias, padding=weight.shape[-1] // 2)

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="mps")
    def backward(ctx, output_grad):
        x, weight = ctx.saved_tensors
        output_grad = output_grad.contiguous()
        kernel = weight.shape[-1]
        input_grad = weight_grad = bias_grad = None
        if ctx.needs_input_grad[0]:
            # Same result as the transposed convolution, which MPS runs twice as slowly.
            flipped_weight = weight.flip((2, 3, 4)).transpose(0, 1)
            input_grad = F.conv3d(output_grad, flipped_weight, padding=kernel // 2)
        if ctx.needs_input_grad[1]:
            weight_grad = compute_weight_gradient(x, output_grad, kernel)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            bias_grad = output_grad.sum((0, 2, 3, 4), dtype=torch.float32)
        return input_grad, weight_grad, bias_grad


class Conv3d(nn.Conv3d):
    def _conv_forward(self, x, weight, bias):
        # Anchor guidance needs only input gradients of frozen weights.
        needs_gradient = torch.is_grad_enabled() and (
            weight.requires_grad or x.requires_grad
        )
        is_plain = (
            self.stride == (1, 1, 1)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
            and self.padding_mode == "zeros"
        )
        if not (x.device.type == "mps" and needs_gradient and is_plain):
            return super()._conv_forward(x, weight, bias)
        if self.kernel_size == (1, 1, 1) and self.padding == (0, 0, 0):
            # Summing pointwise gradients over the whole grid can overflow fp16.
            with torch.autocast("mps", enabled=False):
                out = torch.matmul(weight.flatten(1), x.flatten(2).float())
            if bias is not None:
                out = out + bias[:, None]
            return out.view(x.shape[0], -1, *x.shape[2:])
        if (
            self.kernel_size == (3, 3, 3)
            and self.padding == (1, 1, 1)
            and x.shape[2:].numel() >= MIN_VOXELS
        ):
            return _Conv3dFunction.apply(x, weight, bias)
        return super()._conv_forward(x, weight, bias)


def compute_weight_gradient(x, output_grad, kernel):
    batch, channels, depth, height, width = x.shape
    out_channels = output_grad.shape[1]
    pad = kernel // 2
    plane_taps = kernel * kernel
    pixels = height * width
    grad_channels_last = output_grad.permute(0, 2, 3, 4, 1)
    slice_bytes = batch * channels * plane_taps * pixels * x.element_size()
    slices_per_chunk = max(1, CHUNK_BYTES // slice_bytes - 2 * pad)
    dtype = torch.promote_types(x.dtype, torch.float32)
    weight_grad = x.new_zeros(kernel, channels * plane_taps, out_channels, dtype=dtype)
    for start in range(0, depth, slices_per_chunk):
        size = min(slices_per_chunk, depth - start)
        low, high = max(start - pad, 0), min(start + size + pad, depth)
        missing_before, missing_after = low - (start - pad), start + size + pad - high
        padded = F.pad(
            x[:, :, low:high], (0, 0, 0, 0, missing_before, missing_after)
        ).transpose(1, 2)
        columns = F.unfold(
            padded.reshape(-1, channels, height, width), kernel, padding=pad
        ).view(batch, size + 2 * pad, channels * plane_taps, pixels)
        grad_chunk = grad_channels_last[:, start : start + size].reshape(
            batch * size, pixels, out_channels
        )
        for depth_tap in range(kernel):
            tap_columns = columns[:, depth_tap : depth_tap + size].reshape(
                batch * size, channels * plane_taps, pixels
            )
            weight_grad[depth_tap] += torch.bmm(tap_columns, grad_chunk).sum(
                0, dtype=dtype
            )
    return weight_grad.view(kernel, channels, kernel, kernel, out_channels).permute(
        4, 1, 0, 2, 3
    )
