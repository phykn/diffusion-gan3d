from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AdaptiveResBlock2D,
    SinusoidalTimeEmbedding,
    group_count,
)


class PairCritic2D(nn.Module):
    """Scores a time-conditioned `(x_previous, x_current)` slice pair."""

    def __init__(
        self,
        *,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(num_phases, int) or num_phases < 2:
            raise ValueError("num_phases must be an integer of at least 2.")
        if not isinstance(embedding_channels, int) or embedding_channels < 4:
            raise ValueError("embedding_channels must be an integer of at least 4.")
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")
        hidden_channels = _positive_channels(channels)

        self.num_phases = num_phases
        self.downsample_factor = 2 ** (len(hidden_channels) - 1)
        self.gradient_checkpointing = gradient_checkpointing
        self.time_embedding = SinusoidalTimeEmbedding(embedding_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.input = nn.Conv2d(2 * num_phases, hidden_channels[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            AdaptiveResBlock2D(channel, channel, embedding_channels)
            for channel in hidden_channels
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(
                hidden_channels[index],
                hidden_channels[index + 1],
                3,
                stride=2,
                padding=1,
            )
            for index in range(len(hidden_channels) - 1)
        )
        self.output_norm = nn.GroupNorm(
            group_count(hidden_channels[-1]),
            hidden_channels[-1],
        )
        self.output = nn.Linear(hidden_channels[-1], 1)

    def forward(
        self,
        x_previous: torch.Tensor,
        x_current: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(x_previous, x_current, time)
        embedding = self.time_mlp(
            self.time_embedding(time.to(device=x_previous.device)).to(
                dtype=x_previous.dtype
            )
        )
        hidden = self.input(torch.cat((x_previous, x_current), dim=1))
        for index, block in enumerate(self.blocks):
            hidden = self._block(block, hidden, embedding)
            if index < len(self.downsample):
                hidden = self.downsample[index](hidden)
        hidden = F.silu(self.output_norm(hidden)).mean(dim=(-2, -1))
        return self.output(hidden).squeeze(1)

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
        x_previous: torch.Tensor,
        x_current: torch.Tensor,
        time: torch.Tensor,
    ) -> None:
        if not isinstance(x_previous, torch.Tensor) or x_previous.ndim != 4:
            raise ValueError("x_previous must have shape [B, P, H, W].")
        if not isinstance(x_current, torch.Tensor) or x_current.ndim != 4:
            raise ValueError("x_current must have shape [B, P, H, W].")
        if x_previous.shape != x_current.shape:
            raise ValueError("x_previous and x_current must have the same shape.")
        if not x_previous.is_floating_point() or not x_current.is_floating_point():
            raise ValueError("slice pairs must be floating point.")
        if x_previous.shape[1] != self.num_phases:
            raise ValueError("slice pairs have the wrong number of phase channels.")
        if any(size % self.downsample_factor for size in x_previous.shape[-2:]):
            raise ValueError(
                "every spatial size must be divisible by "
                f"{self.downsample_factor}."
            )
        if not isinstance(time, torch.Tensor) or time.shape != (
            x_previous.shape[0],
        ):
            raise ValueError("time must have shape [B].")


def _positive_channels(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("channels must be a non-empty sequence.")
    channels = tuple(values)
    if not channels:
        raise ValueError("channels must be a non-empty sequence.")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in channels
    ):
        raise ValueError("channels must contain positive integers.")
    return channels
