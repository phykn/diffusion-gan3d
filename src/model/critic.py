from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.device import adaptive_pool
from src.model.layers import (
    AdaptiveResBlock,
    SinusoidalEmbedding,
    build_embedding_mlp,
    run_block,
)

MAX_NORM_GROUPS = 32


@dataclass(frozen=True)
class CriticScores:
    logits_global: torch.Tensor
    levels: tuple = ()


class GroupNormSiLU(nn.GroupNorm):
    def __init__(self, channels, affine=True):
        super().__init__(self._choose_groups(channels), channels, affine=affine)

    def forward(self, x, scale=None, shift=None):
        x = super().forward(x)
        if scale is not None:
            x = x * scale + shift
        return F.silu(x)

    @staticmethod
    def _choose_groups(channels):
        most = min(MAX_NORM_GROUPS, max(channels // 2, 1))
        for groups in range(most, 0, -1):
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
            AdaptiveResBlock(GroupNormSiLU, nn.Conv2d, width, embedding_channels)
            for width in widths
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(narrow, wide, 3, stride=2, padding=1)
            for narrow, wide in zip(widths[:-1], widths[1:], strict=True)
        )
        self.output_norm = GroupNormSiLU(widths[-1])
        self.output = nn.Linear(widths[-1], 1)
        self.time_embedding = SinusoidalEmbedding(embedding_channels)
        self.time_mlp = build_embedding_mlp(embedding_channels, embedding_channels)

    def forward(self, x_previous, x_current, time, height=None):
        pair = torch.cat((x_previous, x_current), dim=1)
        embedding = self.time_mlp(
            self.time_embedding(time.to(device=pair.device)).to(dtype=pair.dtype)
        )
        scores = tuple(
            self._score_one_size(resized, embedding, height)
            for resized in self._build_pyramid(pair)
        )
        return CriticScores(scores[0].logits_global, scores if len(scores) > 1 else ())

    def _build_pyramid(self, pair):
        yield pair
        while min(pair.shape[-2:]) >= 2 * self.pyramid_min_size:
            pair = adaptive_pool(pair, tuple(size // 2 for size in pair.shape[-2:]))
            yield pair

    def _score_one_size(self, pair, embedding, height):
        x = self.input(pair)
        # xy sections have no height field: they are scored without one.
        if self.height_input is not None and height is not None:
            x = x + self.height_input(adaptive_pool(height.to(pair), pair.shape[-2:]))
        checkpointing = self.gradient_checkpointing and self.training
        for level, block in enumerate(self.blocks):
            x = run_block(block, x, embedding, checkpointing)
            if level < len(self.downsample):
                x = self.downsample[level](x)
        x = self.output_norm(x).mean(dim=(-2, -1))
        return CriticScores(logits_global=self.output(x).squeeze(1))
