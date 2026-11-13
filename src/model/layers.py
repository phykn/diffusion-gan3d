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
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / denom
        )
        self.channels = channels
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, value):
        angles = value.to(dtype=torch.float32)[:, None] * self.freqs[None]
        emb = torch.cat((angles.cos(), angles.sin()), dim=1)
        if emb.shape[1] < self.channels:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaptiveNorm(nn.Module):
    def __init__(self, norm, channels, embedding_channels):
        super().__init__()
        self.norm = norm(channels, affine=False)
        self.affine = nn.Linear(embedding_channels, 2 * channels)

    def forward(self, x, emb):
        scale, shift = self.affine(F.silu(emb)).chunk(2, dim=1)
        dims = (1,) * (x.ndim - 2)
        scale = scale.reshape(scale.shape + dims)
        shift = shift.reshape(shift.shape + dims)
        return self.norm(x, 1.0 + scale, shift)


class AdaptiveResBlock(nn.Module):
    def __init__(self, norm, conv, channels, embedding_channels):
        super().__init__()
        self.norm1 = AdaptiveNorm(norm, channels, embedding_channels)
        self.conv1 = conv(channels, channels, 3, padding=1)
        self.norm2 = AdaptiveNorm(norm, channels, embedding_channels)
        self.conv2 = conv(channels, channels, 3, padding=1)

    def forward(self, x, emb):
        h = self.conv1(self.norm1(x, emb))
        h = self.conv2(self.norm2(h, emb))
        return (x + h) * INV_SQRT_TWO


def run_block(block, inputs, emb, checkpointing):
    if checkpointing and torch.is_grad_enabled():
        return checkpoint(block, inputs, emb, use_reentrant=False)
    return block(inputs, emb)


def embedding_mlp(in_channels, channels):
    return nn.Sequential(
        nn.Linear(in_channels, channels),
        nn.SiLU(),
        nn.Linear(channels, channels),
    )
