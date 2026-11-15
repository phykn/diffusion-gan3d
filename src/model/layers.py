import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        steps = torch.arange(half, dtype=torch.float32)
        last_step = max(half - 1, 1)
        frequencies = torch.exp(-math.log(10_000.0) * steps / last_step)
        self.channels = channels
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, value):
        angles = value.to(dtype=torch.float32)[:, None] * self.frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
        if embedding.shape[1] < self.channels:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class AdaptiveNorm(nn.Module):
    def __init__(self, norm, channels, embedding_channels):
        super().__init__()
        self.norm = norm(channels, affine=False)
        self.affine = nn.Linear(embedding_channels, 2 * channels)

    def forward(self, x, embedding):
        scale, shift = self.affine(F.silu(embedding)).chunk(2, dim=1)
        spatial_ones = (1,) * (x.ndim - 2)
        scale = scale.reshape(scale.shape + spatial_ones)
        shift = shift.reshape(shift.shape + spatial_ones)
        return self.norm(x, 1.0 + scale, shift)


class AdaptiveResBlock(nn.Module):
    def __init__(self, norm, conv, channels, embedding_channels):
        super().__init__()
        self.norm1 = AdaptiveNorm(norm, channels, embedding_channels)
        self.conv1 = conv(channels, channels, 3, padding=1)
        self.norm2 = AdaptiveNorm(norm, channels, embedding_channels)
        self.conv2 = conv(channels, channels, 3, padding=1)

    def forward(self, x, embedding):
        residual = self.conv1(self.norm1(x, embedding))
        residual = self.conv2(self.norm2(residual, embedding))
        return (x + residual) * INV_SQRT_TWO


def run_block(block, x, embedding, checkpointing):
    if checkpointing and torch.is_grad_enabled():
        return checkpoint(block, x, embedding, use_reentrant=False)
    return block(x, embedding)


def build_embedding_mlp(in_channels, channels):
    return nn.Sequential(
        nn.Linear(in_channels, channels),
        nn.SiLU(),
        nn.Linear(channels, channels),
    )
