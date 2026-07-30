import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AdaptiveResBlock3D,
    Downsample3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
    group_count,
)

_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


class Denoiser3D(nn.Module):
    """Time- and latent-conditioned 3D U-Net for clean categorical samples."""

    def __init__(
        self,
        *,
        num_phases: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        embedding_channels: int,
        latent_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(num_phases, int) or num_phases < 2:
            raise ValueError("num_phases must be an integer of at least 2.")
        if not isinstance(base_channels, int) or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer.")
        if not isinstance(embedding_channels, int) or embedding_channels < 4:
            raise ValueError("embedding_channels must be an integer of at least 4.")
        if not isinstance(latent_channels, int) or latent_channels <= 0:
            raise ValueError("latent_channels must be a positive integer.")
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")
        multipliers = _positive_channels(channel_multipliers)

        channels = tuple(base_channels * multiplier for multiplier in multipliers)
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** (len(channels) - 1)
        self.gradient_checkpointing = gradient_checkpointing

        self.time_embedding = SinusoidalTimeEmbedding(embedding_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.input = nn.Conv3d(num_phases, channels[0], 3, padding=1)

        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.encoder.append(
                AdaptiveResBlock3D(channel, channel, embedding_channels)
            )
            if index + 1 < len(channels):
                self.downsample.append(
                    nn.Sequential(
                        Downsample3D(channel),
                        nn.Conv3d(channel, channels[index + 1], 1),
                    )
                )

        self.middle = nn.ModuleList(
            (
                AdaptiveResBlock3D(
                    channels[-1],
                    channels[-1],
                    embedding_channels,
                ),
                AdaptiveResBlock3D(
                    channels[-1],
                    channels[-1],
                    embedding_channels,
                ),
            )
        )

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.upsample.append(Upsample3D(current, skip_channels))
            self.decoder.append(
                AdaptiveResBlock3D(
                    2 * skip_channels,
                    skip_channels,
                    embedding_channels,
                )
            )
            current = skip_channels

        self.output_norm = nn.GroupNorm(group_count(channels[0]), channels[0])
        self.output = nn.Conv3d(channels[0], num_phases, 3, padding=1)

    def forward(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(x_current, time, latent)
        embedding = self._conditioning(x_current, time, latent)
        hidden = self.input(x_current)
        skips = []
        for index, block in enumerate(self.encoder):
            hidden = self._block(block, hidden, embedding)
            if index < len(self.downsample):
                skips.append(hidden)
                hidden = self.downsample[index](hidden)

        for block in self.middle:
            hidden = self._block(block, hidden, embedding)

        for upsample, block, skip in zip(
            self.upsample,
            self.decoder,
            reversed(skips),
            strict=True,
        ):
            hidden = upsample(hidden, size=skip.shape[-3:])
            hidden = torch.cat(
                (hidden * _INV_SQRT_TWO, skip * _INV_SQRT_TWO),
                dim=1,
            )
            hidden = self._block(block, hidden, embedding)

        logits = self.output(F.silu(self.output_norm(hidden)))
        return 2.0 * logits.softmax(dim=1) - 1.0

    def _conditioning(
        self,
        inputs: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        time = time.to(device=inputs.device)
        latent = latent.to(device=inputs.device, dtype=inputs.dtype)
        time_embedding = self.time_mlp(
            self.time_embedding(time).to(dtype=inputs.dtype)
        )
        latent_embedding = self.latent_mlp(latent)
        return (time_embedding + latent_embedding) * _INV_SQRT_TWO

    def _block(
        self,
        block: nn.Module,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                block,
                inputs,
                embedding,
                use_reentrant=False,
            )
        return block(inputs, embedding)

    def _validate_inputs(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
    ) -> None:
        if not isinstance(x_current, torch.Tensor) or x_current.ndim != 5:
            raise ValueError("x_current must have shape [B, P, D, H, W].")
        if not x_current.is_floating_point():
            raise ValueError("x_current must be floating point.")
        if x_current.shape[1] != self.num_phases:
            raise ValueError("x_current has the wrong number of phase channels.")
        if any(size % self.downsample_factor for size in x_current.shape[-3:]):
            raise ValueError(
                "every spatial size must be divisible by "
                f"{self.downsample_factor}."
            )
        if not isinstance(time, torch.Tensor) or time.shape != (x_current.shape[0],):
            raise ValueError("time must have shape [B].")
        if not isinstance(latent, torch.Tensor) or latent.shape != (
            x_current.shape[0],
            self.latent_channels,
        ):
            raise ValueError(
                f"latent must have shape [B, {self.latent_channels}]."
            )
        if not latent.is_floating_point():
            raise ValueError("latent must be floating point.")


def _positive_channels(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("channel_multipliers must be a non-empty sequence.")
    channels = tuple(values)
    if not channels:
        raise ValueError("channel_multipliers must be a non-empty sequence.")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in channels
    ):
        raise ValueError("channel_multipliers must contain positive integers.")
    return channels
