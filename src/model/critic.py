from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.device import adaptive_pool
from src.model.layers import (
    AdaptiveResBlock,
    SinusoidalEmbedding,
    embedding_mlp,
    run_block,
)


@dataclass(frozen=True)
class CriticScores:
    logits_global: torch.Tensor
    levels: tuple["CriticScores", ...] = ()


def area_pyramid(images: torch.Tensor, min_size: int = 16) -> Iterator[torch.Tensor]:
    yield images
    while min(images.shape[-2:]) >= 2 * min_size:
        images = adaptive_pool(images, tuple(n // 2 for n in images.shape[-2:]))
        yield images


class GroupNormSiLU(nn.GroupNorm):
    """Group-normalize, scale and shift when given, then apply SiLU."""

    def __init__(self, channels: int, affine: bool = True) -> None:
        super().__init__(self.choose_groups(channels), channels, affine=affine)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x)
        if scale is not None:
            x = x * scale + shift
        return F.silu(x)

    @staticmethod
    def choose_groups(channels: int) -> int:
        limit = min(32, max(channels // 2, 1))
        for groups in range(limit, 0, -1):
            if channels % groups == 0:
                return groups


class PairCritic2D(nn.Module):
    """Critic of adjacent diffusion states (x_t, x_t+1) on 2D sections.

    Each level of an area pyramid is scored with the same weights.
    """

    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        gradient_checkpointing: bool = False,
        *,
        height_channels: int = 0,
        pyramid_min_size: int = 16,
    ) -> None:
        """height_channels adds an optional condition input (0 disables)."""
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2:
            raise ValueError("channels must contain at least two levels.")

        self.gradient_checkpointing = gradient_checkpointing
        self.pyramid_min_size = pyramid_min_size
        self.height_input = (
            nn.Conv2d(height_channels, widths[0], 3, padding=1, bias=False)
            if height_channels
            else None
        )
        self.input = nn.Conv2d(2 * num_phases, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            AdaptiveResBlock(GroupNormSiLU, nn.Conv2d, ch, ch, embedding_channels)
            for ch in widths
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(widths[idx], widths[idx + 1], 3, stride=2, padding=1)
            for idx in range(len(widths) - 1)
        )
        self.output_norm = GroupNormSiLU(widths[-1])
        self.output = nn.Linear(widths[-1], 1)
        self.time_embedding = SinusoidalEmbedding(embedding_channels)
        self.time_mlp = embedding_mlp(embedding_channels, embedding_channels)

    def forward(
        self,
        x_previous: torch.Tensor,
        x_current: torch.Tensor,
        time: torch.Tensor,
        height: torch.Tensor | None = None,
    ) -> CriticScores:
        inputs = torch.cat((x_previous, x_current), dim=1)
        return self.score(inputs, self.embed(time, inputs), height)

    def embed(self, time: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(
            self.time_embedding(time.to(device=ref.device)).to(dtype=ref.dtype)
        )

    def score(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
        height: torch.Tensor | None = None,
    ) -> CriticScores:
        scores = tuple(
            self.score_level(level, embedding, height)
            for level in area_pyramid(inputs, self.pyramid_min_size)
        )
        return CriticScores(scores[0].logits_global, scores if len(scores) > 1 else ())

    def score_level(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
        height: torch.Tensor | None = None,
    ) -> CriticScores:
        x = self.input(inputs)
        # xy sections have no height field: they are scored without one.
        if self.height_input is not None and height is not None:
            x = x + self.height_input(
                adaptive_pool(height.to(inputs), inputs.shape[-2:])
            )
        checkpointing = self.gradient_checkpointing and self.training
        for idx, block in enumerate(self.blocks):
            x = run_block(block, x, embedding, checkpointing)
            if idx < len(self.downsample):
                x = self.downsample[idx](x)
        x = self.output_norm(x).mean(dim=(-2, -1))
        return CriticScores(logits_global=self.output(x).squeeze(1))
