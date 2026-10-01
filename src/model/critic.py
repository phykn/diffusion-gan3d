from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.layers import (
    INV_SQRT_TWO,
    AdaptiveNorm,
    SinusoidalEmbedding,
    embed_domain,
)
from src.model.pyramid import area_pyramid


@dataclass(frozen=True)
class CriticScores:
    logits_global: torch.Tensor
    logits_local: torch.Tensor
    levels: tuple["CriticScores", ...] = ()


class GroupNorm(nn.GroupNorm):
    def __init__(self, channels: int, affine: bool = True) -> None:
        super().__init__(self.choose_groups(channels), channels, affine=affine)

    @staticmethod
    def choose_groups(channels: int, maximum: int = 32) -> int:
        limit = min(maximum, max(channels // 2, 1))
        for groups in range(limit, 0, -1):
            if channels % groups == 0:
                return groups


class AdaptiveResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = AdaptiveNorm(GroupNorm, in_channels, embedding_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveNorm(GroupNorm, out_channels, embedding_channels)
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


class CriticBase(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        embedding_channels: int,
        num_domains: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2:
            raise ValueError("channels must contain at least two levels.")

        self.gradient_checkpointing = gradient_checkpointing
        self.pyramid_min_size = 16
        self.height_input = None
        self.profile_input = None
        self.domain_embedding = nn.Embedding(num_domains, embedding_channels)
        self.input = nn.Conv2d(input_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            AdaptiveResBlock2D(ch, ch, embedding_channels) for ch in widths
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
        self.local_norm = GroupNorm(widths[1])
        self.local_output = nn.Conv2d(widths[1], 1, 1)
        self.output_norm = GroupNorm(widths[-1])
        self.output = nn.Linear(widths[-1], 1)

    def score(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> CriticScores:
        domain_emb = embed_domain(self.domain_embedding, domain, inputs.dtype)
        embedding = (embedding + domain_emb) * INV_SQRT_TWO
        scores = tuple(
            self.score_level(level, embedding, height, profile)
            for level in area_pyramid(inputs, self.pyramid_min_size)
        )
        return CriticScores(
            scores[0].logits_global,
            scores[0].logits_local,
            scores if len(scores) > 1 else (),
        )

    def score_level(self, inputs, embedding, height=None, profile=None) -> CriticScores:
        x = self.input(inputs)
        if height is not None and self.height_input is not None:
            x = x + self.height_input(
                F.interpolate(height.to(inputs), size=inputs.shape[-2:], mode="area")
            )
        if profile is not None and self.profile_input is not None:
            x = x + self.profile_input(
                F.interpolate(profile.to(inputs), size=inputs.shape[-2:], mode="area")
            )
        for idx, block in enumerate(self.blocks):
            x = self.apply_block(block, x, embedding)
            if idx == 1:
                local = F.silu(self.local_norm(x))
                logits_local = self.local_output(local).squeeze(1)
            if idx < len(self.downsample):
                x = self.downsample[idx](x)
        x = F.silu(self.output_norm(x)).mean(dim=(-2, -1))
        logits_global = self.output(x).squeeze(1)
        return CriticScores(
            logits_global=logits_global,
            logits_local=logits_local,
        )

    def apply_block(
        self,
        block: nn.Module,
        inputs: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                block,
                inputs,
                emb,
                use_reentrant=False,
            )
        return block(inputs, emb)


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
    ) -> None:
        super().__init__(
            input_channels=num_phases,
            channels=channels,
            embedding_channels=embedding_channels,
            num_domains=num_domains,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.time_embedding = SinusoidalEmbedding(embedding_channels)
        self.time_scale = time_scale
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )

    def forward(
        self,
        x_previous: torch.Tensor,
        time: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> CriticScores:
        embedding = self.time_mlp(
            self.time_embedding(time.to(device=x_previous.device) * self.time_scale).to(
                dtype=x_previous.dtype
            )
        )
        return self.score(
            x_previous,
            embedding,
            domain,
            height,
            profile,
        )
