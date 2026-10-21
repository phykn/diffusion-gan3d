from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.device import adaptive_pool, assert_all
from src.model.layers import (
    INV_SQRT_TWO,
    AdaptiveResBlock,
    SinusoidalEmbedding,
    embed_domain,
    embedding_mlp,
    run_block,
)
from src.model.pyramid import area_pyramid


@dataclass(frozen=True)
class CriticScores:
    logits_global: torch.Tensor
    logits_local: torch.Tensor | None = None
    levels: tuple["CriticScores", ...] = ()


def _select(scores: CriticScores, part: slice) -> CriticScores:
    return CriticScores(
        scores.logits_global[part],
        None if scores.logits_local is None else scores.logits_local[part],
        tuple(_select(level, part) for level in scores.levels),
    )


class GroupNorm(nn.GroupNorm):
    def __init__(self, channels: int, affine: bool = True) -> None:
        super().__init__(self.choose_groups(channels), channels, affine=affine)

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
        local_head: bool = False,
        *,
        height_channels: int = 0,
        profile_channels: int = 0,
        pyramid_min_size: int = 16,
    ) -> None:
        """height/profile_channels add optional condition inputs (0 disables)."""
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2:
            raise ValueError("channels must contain at least two levels.")

        self.gradient_checkpointing = gradient_checkpointing
        self.pyramid_min_size = pyramid_min_size
        self.height_input = self._condition_input(height_channels, widths[0])
        self.profile_input = self._condition_input(profile_channels, widths[0])
        self.domain_embedding = nn.Embedding(num_domains, embedding_channels)
        self.input = nn.Conv2d(input_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            AdaptiveResBlock(GroupNorm, nn.Conv2d, ch, ch, embedding_channels)
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
        self.local_norm = GroupNorm(widths[1]) if local_head else None
        self.local_output = nn.Conv2d(widths[1], 1, 1) if local_head else None
        self.output_norm = GroupNorm(widths[-1])
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
        logits_local = None
        if height is not None and self.height_input is not None:
            x = x + self.height_input(
                adaptive_pool(height.to(inputs), inputs.shape[-2:])
            )
        if profile is not None and self.profile_input is not None:
            x = x + self.profile_input(
                adaptive_pool(profile.to(inputs), inputs.shape[-2:])
            )
        checkpointing = self.gradient_checkpointing and self.training
        for idx, block in enumerate(self.blocks):
            x = run_block(block, x, embedding, checkpointing)
            if idx == 1 and self.local_output is not None:
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
        profile: torch.Tensor | None = None,
    ) -> CriticScores:
        return super().forward(
            torch.cat((x_previous, x_current), dim=1), time, domain, height, profile
        )


class SeamCritic2D(CriticBase):
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
        self.axis_embedding = nn.Embedding(3, embedding_channels)
        self.time_embedding = SinusoidalEmbedding(embedding_channels)
        self.time_scale = time_scale
        self.time_mlp = embedding_mlp(embedding_channels, embedding_channels)

    def forward(
        self,
        bands: torch.Tensor,
        axes: torch.Tensor,
        time: torch.Tensor,
        domain: torch.Tensor,
    ) -> CriticScores:
        if bands.ndim != 4:
            raise ValueError("bands must have shape [B, C, H, W].")
        if axes.shape != (bands.shape[0],):
            raise ValueError("axes must have shape [B].")
        axes = axes.to(device=bands.device, dtype=torch.long)
        axis_embedding = self.axis_embedding(axes).to(dtype=bands.dtype)
        time_embedding = self.time_mlp(
            self.time_embedding(time.to(device=bands.device) * self.time_scale).to(
                dtype=bands.dtype
            )
        )
        embedding = (axis_embedding + time_embedding) * INV_SQRT_TWO
        return self.score(bands, embedding, domain)


class ConnectivityCritic2D(CriticBase):
    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        num_domains: int,
        gradient_checkpointing: bool = False,
        directed_axis: int | None = None,
        **inputs,
    ) -> None:
        super().__init__(
            input_channels=3 * num_phases,
            channels=channels,
            embedding_channels=embedding_channels,
            num_domains=num_domains,
            gradient_checkpointing=gradient_checkpointing,
            local_head=True,
            **inputs,
        )
        self.axis_embedding = nn.Embedding(3, embedding_channels)
        self.directed_axis = directed_axis
        self.gap_embedding = SinusoidalEmbedding(embedding_channels)
        self.gap_mlp = embedding_mlp(embedding_channels, embedding_channels)

    def forward(
        self,
        triplets: torch.Tensor,
        axes: torch.Tensor,
        gaps: torch.Tensor,
        domain: torch.Tensor,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> CriticScores:
        if axes.shape != (triplets.shape[0],):
            raise ValueError("axes must have shape [B].")
        if gaps.shape != (triplets.shape[0],):
            raise ValueError("gaps must have shape [B].")
        axes = axes.to(device=triplets.device, dtype=torch.long)
        gaps = gaps.to(device=triplets.device, dtype=torch.float32)
        valid = ((axes >= 0) & (axes <= 2) & (gaps >= 1)).all()
        assert_all(
            valid, "axes must contain only 0, 1, or 2 and gaps must be positive."
        )

        # Score both slab orders in one pass; normalization is per sample.
        both = self.score_once(
            torch.cat((triplets, triplets.flip(1))),
            axes.repeat(2),
            gaps.repeat(2),
            domain.repeat(2),
            None if height is None else torch.cat((height, height.flip(1))),
            None if profile is None else torch.cat((profile, profile.flip(1))),
        )
        count = len(axes)
        forward = _select(both, slice(None, count))
        reverse = _select(both, slice(count, None))
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
