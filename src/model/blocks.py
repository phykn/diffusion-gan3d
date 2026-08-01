import math

import torch
import torch.nn.functional as F
from torch import nn

INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


def choose_groups(channels: int, maximum: int = 32) -> int:
    limit = min(maximum, max(channels // 2, 1))
    for groups in range(limit, 0, -1):
        if channels % groups == 0:
            return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        half = channels // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / denom
        )
        self.channels = channels
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if not isinstance(time, torch.Tensor) or time.ndim != 1:
            raise ValueError("time must have shape [B].")
        angles = time.to(dtype=torch.float32)[:, None] * self.freqs[None]
        emb = torch.cat((angles.cos(), angles.sin()), dim=1)
        if emb.shape[1] < self.channels:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, channels: int, embedding_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(
            choose_groups(channels),
            channels,
            affine=False,
        )
        self.affine = nn.Linear(embedding_channels, 2 * channels)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.affine(F.silu(emb)).chunk(2, dim=1)
        dims = (1,) * (x.ndim - 2)
        scale = scale.reshape(scale.shape + dims)
        shift = shift.reshape(shift.shape + dims)
        return self.norm(x) * (1.0 + scale) + shift


class ChannelNorm3D(nn.Module):
    def __init__(
        self,
        channels: int,
        eps: float = 1.0e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("scale", None)
            self.register_parameter("shift", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(
            x,
            dim=1,
            keepdim=True,
            correction=0,
        )
        mean = mean.to(dtype=x.dtype)
        inv_std = torch.rsqrt(var + self.eps).to(dtype=x.dtype)
        x = (x - mean) * inv_std
        if self.scale is None:
            return x
        shape = (1, -1, 1, 1, 1)
        scale = self.scale.to(dtype=x.dtype).view(shape)
        shift = self.shift.to(dtype=x.dtype).view(shape)
        return x * scale + shift


class AdaptiveChannelNorm3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int) -> None:
        super().__init__()
        self.norm = ChannelNorm3D(channels, affine=False)
        self.affine = nn.Linear(embedding_channels, 2 * channels)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.affine(F.silu(emb)).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return self.norm(x) * (1.0 + scale) + shift


class AdaptiveResBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = AdaptiveChannelNorm3D(in_channels, embedding_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveChannelNorm3D(out_channels, embedding_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, emb)))
        h = self.conv2(F.silu(self.norm2(h, emb)))
        return (self.skip(x) + h) * INV_SQRT_TWO


class AdaptiveResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_channels, embedding_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(out_channels, embedding_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, emb)))
        h = self.conv2(F.silu(self.norm2(h, emb)))
        return (self.skip(x) + h) * INV_SQRT_TWO


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        h = F.interpolate(x, size=size, mode="nearest")
        return self.conv(h)
