import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


def choose_groups(channels: int, *, maximum: int = 32) -> int:
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ValueError("channels must be a positive integer.")
    limit = min(maximum, max(channels // 2, 1))
    for groups in range(limit, 0, -1):
        if channels % groups == 0:
            return groups


def check_channels(
    name: str,
    values: Sequence[int],
    *,
    minimum: int = 1,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence.")
    channels = tuple(values)
    if not channels:
        raise ValueError(f"{name} must be a non-empty sequence.")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < minimum
        for value in channels
    ):
        raise ValueError(f"{name} must contain integers of at least {minimum}.")
    return channels


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels: int, *, max_period: float = 10_000.0) -> None:
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 2:
            raise ValueError("channels must be an integer of at least 2.")
        if not math.isfinite(max_period) or max_period <= 1.0:
            raise ValueError("max_period must be finite and greater than 1.")

        half = channels // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32)
            / denominator
        )
        self.channels = channels
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if not isinstance(time, torch.Tensor) or time.ndim != 1:
            raise ValueError("time must have shape [B].")
        angles = time.to(dtype=torch.float32)[:, None] * self.frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
        if embedding.shape[1] < self.channels:
            embedding = F.pad(embedding, (0, 1))
        return embedding


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
        inputs: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[0] != inputs.shape[0]:
            raise ValueError("embedding must have shape [B, E].")
        scale, shift = self.affine(F.silu(embedding)).chunk(2, dim=1)
        trailing = (1,) * (inputs.ndim - 2)
        scale = scale.reshape(scale.shape + trailing)
        shift = shift.reshape(shift.shape + trailing)
        return self.norm(inputs) * (1.0 + scale) + shift


class AdaptiveResBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_channels, embedding_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(out_channels, embedding_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs, embedding)))
        hidden = self.conv2(F.silu(self.norm2(hidden, embedding)))
        return (self.skip(inputs) + hidden) * _INV_SQRT_TWO


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
        inputs: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs, embedding)))
        hidden = self.conv2(F.silu(self.norm2(hidden, embedding)))
        return (self.skip(inputs) + hidden) * _INV_SQRT_TWO


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden = F.interpolate(inputs, size=size, mode="nearest")
        return self.conv(hidden)
