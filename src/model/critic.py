from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.device import adaptive_pool
from src.model.layers import (
    INV_SQRT_TWO,
    AdaptiveResBlock,
    SinusoidalEmbedding,
    embed_domain,
    embedding_mlp,
    run_block,
)


@dataclass(frozen=True)
class CriticScores:
    logits_global: torch.Tensor
    levels: tuple["CriticScores", ...] = ()


def area_pyramid(images, min_size=16):
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


class CriticBase(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        embedding_channels: int,
        num_domains: int,
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
        self.height_input = self._condition_input(height_channels, widths[0])
        self.domain_embedding = nn.Embedding(num_domains, embedding_channels)
        self.input = nn.Conv2d(input_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            AdaptiveResBlock(GroupNormSiLU, nn.Conv2d, ch, ch, embedding_channels)
            for ch in widths
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(
                widths[idx],
                widths[idx + 1],
                3,
                stride=2,
                padding=1,
            )
            for idx in range(len(widths) - 1)
        )
        self.output_norm = GroupNormSiLU(widths[-1])
        self.output = nn.Linear(widths[-1], 1)

    @staticmethod
    def _condition_input(channels: int, width: int) -> nn.Conv2d | None:
        if not channels:
            return None
        return nn.Conv2d(channels, width, 3, padding=1, bias=False)

    def score(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
    ) -> CriticScores:
        domain_emb = embed_domain(self.domain_embedding, domain, inputs.dtype)
        embedding = (embedding + domain_emb) * INV_SQRT_TWO
        scores = tuple(
            self.score_level(level, embedding, height)
            for level in area_pyramid(inputs, self.pyramid_min_size)
        )
        return CriticScores(scores[0].logits_global, scores if len(scores) > 1 else ())

    def score_level(self, inputs, embedding, height=None) -> CriticScores:
        x = self.input(inputs)
        if height is not None and self.height_input is not None:
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


class PlaneCritic2D(CriticBase):
    input_mode = "single"

    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        num_domains: int,
        gradient_checkpointing: bool = False,
        time_scale: float = 1.0,
        **inputs,
    ) -> None:
        super().__init__(
            input_channels=num_phases,
            channels=channels,
            embedding_channels=embedding_channels,
            num_domains=num_domains,
            gradient_checkpointing=gradient_checkpointing,
            **inputs,
        )
        self.time_embedding = SinusoidalEmbedding(embedding_channels)
        self.time_scale = time_scale
        self.time_mlp = embedding_mlp(embedding_channels, embedding_channels)

    def forward(
        self,
        x_previous: torch.Tensor,
        time: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
    ) -> CriticScores:
        embedding = self.time_mlp(
            self.time_embedding(time.to(device=x_previous.device) * self.time_scale).to(
                dtype=x_previous.dtype
            )
        )
        return self.score(x_previous, embedding, domain, height)


class PairCritic2D(PlaneCritic2D):
    """Joint critic for adjacent diffusion states."""

    input_mode = "pair"

    def __init__(self, num_phases: int, *args, **kwargs) -> None:
        super().__init__(2 * num_phases, *args, **kwargs)

    def forward(
        self,
        x_previous: torch.Tensor,
        x_current: torch.Tensor,
        time: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
    ) -> CriticScores:
        return super().forward(
            torch.cat((x_previous, x_current), dim=1), time, domain, height
        )
