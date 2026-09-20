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


class PairCritic2D(CriticBase):
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
            input_channels=2 * num_phases,
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
        x_current: torch.Tensor,
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
            torch.cat((x_previous, x_current), dim=1),
            embedding,
            domain,
            height,
            profile,
        )


class ConnectivityCritic2D(CriticBase):
    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        num_domains: int,
        gradient_checkpointing: bool = False,
        directed_axis: int | None = None,
    ) -> None:
        super().__init__(
            input_channels=3 * num_phases,
            channels=channels,
            embedding_channels=embedding_channels,
            num_domains=num_domains,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.axis_embedding = nn.Embedding(3, embedding_channels)
        self.directed_axis = directed_axis
        self.gap_embedding = SinusoidalEmbedding(embedding_channels)
        self.gap_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )

    def forward(
        self,
        triplets: torch.Tensor,
        axes: torch.Tensor,
        gaps: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> CriticScores:
        if triplets.ndim != 5 or triplets.shape[1] != 3:
            raise ValueError("triplets must have shape [B, 3, C, H, W].")
        if axes.shape != (triplets.shape[0],):
            raise ValueError("axes must have shape [B].")
        if gaps.shape != (triplets.shape[0],):
            raise ValueError("gaps must have shape [B].")
        axes = axes.to(device=triplets.device, dtype=torch.long)
        gaps = gaps.to(device=triplets.device, dtype=torch.float32)
        valid = ((axes >= 0) & (axes <= 2) & (gaps >= 1)).all()
        if axes.device.type == "cuda":
            torch._assert_async(valid, "invalid connectivity axes or gaps")
        elif not bool(valid):
            raise ValueError(
                "axes must contain only 0, 1, or 2 and gaps must be positive."
            )

        forward = self.score_once(triplets, axes, gaps, domain, height, profile)
        reverse = self.score_once(
            triplets.flip(1),
            axes,
            gaps,
            domain,
            None if height is None else height.flip(1),
            None if profile is None else profile.flip(1),
        )
        directed = (
            axes == self.directed_axis
            if self.directed_axis is not None
            else torch.zeros_like(axes, dtype=torch.bool)
        )

        def combine(forward, reverse):
            return CriticScores(
                logits_global=torch.where(
                    directed,
                    forward.logits_global,
                    (forward.logits_global + reverse.logits_global) * 0.5,
                ),
                logits_local=torch.where(
                    directed[:, None, None],
                    forward.logits_local,
                    (forward.logits_local + reverse.logits_local) * 0.5,
                ),
            )

        result = combine(forward, reverse)
        if forward.levels:
            return CriticScores(
                result.logits_global,
                result.logits_local,
                tuple(
                    combine(f, r)
                    for f, r in zip(forward.levels, reverse.levels, strict=True)
                ),
            )
        return result

    def score_once(
        self,
        triplets: torch.Tensor,
        axes: torch.Tensor,
        gaps: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> CriticScores:
        axis_embedding = self.axis_embedding(axes).to(dtype=triplets.dtype)
        gap_embedding = self.gap_mlp(self.gap_embedding(gaps)).to(dtype=triplets.dtype)
        embedding = (axis_embedding + gap_embedding) * INV_SQRT_TWO
        return self.score(
            self.connectivity_images(triplets).flatten(1, 2),
            embedding,
            domain,
            None if height is None else height.flatten(1, 2),
            None if profile is None else profile.flatten(1, 2),
        )

    @staticmethod
    def connectivity_images(triplets: torch.Tensor) -> torch.Tensor:
        if triplets.ndim != 5 or triplets.shape[1] != 3:
            raise ValueError("triplets must have shape [B, 3, C, H, W].")
        phases = (triplets + 1.0) * 0.5
        first = phases[:, 1] - phases[:, 0]
        second = phases[:, 2] - phases[:, 1]
        bend = (second - first) * 0.5
        return torch.stack((first, second, bend), dim=1)
