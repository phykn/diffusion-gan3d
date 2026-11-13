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


def area_pyramid(images, min_size):
    yield images
    while min(images.shape[-2:]) >= 2 * min_size:
        images = adaptive_pool(images, tuple(n // 2 for n in images.shape[-2:]))
        yield images


class GroupNormSiLU(nn.GroupNorm):
    def __init__(self, channels, affine=True):
        super().__init__(self.choose_groups(channels), channels, affine=affine)

    def forward(self, x, scale=None, shift=None):
        x = super().forward(x)
        if scale is not None:
            x = x * scale + shift
        return F.silu(x)

    @staticmethod
    def choose_groups(channels):
        limit = min(32, max(channels // 2, 1))
        for groups in range(limit, 0, -1):
            if channels % groups == 0:
                return groups


class PairCritic2D(nn.Module):
    def __init__(
        self,
        num_phases,
        channels,
        embedding_channels,
        gradient_checkpointing=False,
        *,
        height_channels=0,
        pyramid_min_size=16,
    ):
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
            AdaptiveResBlock(GroupNormSiLU, nn.Conv2d, ch, embedding_channels)
            for ch in widths
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(source, target, 3, stride=2, padding=1)
            for source, target in zip(widths[:-1], widths[1:], strict=True)
        )
        self.output_norm = GroupNormSiLU(widths[-1])
        self.output = nn.Linear(widths[-1], 1)
        self.time_embedding = SinusoidalEmbedding(embedding_channels)
        self.time_mlp = embedding_mlp(embedding_channels, embedding_channels)

    def forward(self, x_previous, x_current, time, height=None):
        inputs = torch.cat((x_previous, x_current), dim=1)
        return self.score(inputs, self.embed(time, inputs), height)

    def embed(self, time, ref):
        return self.time_mlp(
            self.time_embedding(time.to(device=ref.device)).to(dtype=ref.dtype)
        )

    def score(self, inputs, embedding, height=None):
        scores = tuple(
            self.score_level(level, embedding, height)
            for level in area_pyramid(inputs, self.pyramid_min_size)
        )
        return CriticScores(scores[0].logits_global, scores if len(scores) > 1 else ())

    def score_level(self, inputs, embedding, height):
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
