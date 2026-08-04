from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AdaptiveResBlock2D,
    SinusoidalTimeEmbedding,
    choose_groups,
)


@dataclass(frozen=True)
class CriticScores:
    logits_global: torch.Tensor
    logits_local: torch.Tensor


class _Critic2D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        embedding_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2:
            raise ValueError("channels must contain at least two levels.")

        self.gradient_checkpointing = gradient_checkpointing
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
        self.local_norm = nn.GroupNorm(
            choose_groups(widths[1]),
            widths[1],
        )
        self.local_output = nn.Conv2d(widths[1], 1, 1)
        self.output_norm = nn.GroupNorm(
            choose_groups(widths[-1]),
            widths[-1],
        )
        self.output = nn.Linear(widths[-1], 1)

    def score(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
    ) -> CriticScores:
        x = self.input(inputs)
        for idx, block in enumerate(self.blocks):
            x = self.run_block(block, x, embedding)
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

    def run_block(
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


class PairCritic2D(_Critic2D):
    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            input_channels=2 * num_phases,
            channels=channels,
            embedding_channels=embedding_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.time_embedding = SinusoidalTimeEmbedding(embedding_channels)
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
    ) -> CriticScores:
        embedding = self.time_mlp(
            self.time_embedding(time.to(device=x_previous.device)).to(
                dtype=x_previous.dtype
            )
        )
        return self.score(torch.cat((x_previous, x_current), dim=1), embedding)


class ConnectivityCritic2D(_Critic2D):
    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        reversal_invariant: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            input_channels=3 * num_phases,
            channels=channels,
            embedding_channels=embedding_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
        if not isinstance(reversal_invariant, bool):
            raise TypeError("reversal_invariant must be a boolean.")

        self.reversal_invariant = reversal_invariant
        self.axis_embedding = nn.Embedding(3, embedding_channels)
        self.axis_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )

    def forward(
        self,
        triplets: torch.Tensor,
        axes: torch.Tensor,
    ) -> CriticScores:
        if triplets.ndim != 5 or triplets.shape[1] != 3:
            raise ValueError("triplets must have shape [B, 3, C, H, W].")
        if axes.shape != (triplets.shape[0],):
            raise ValueError("axes must have shape [B].")
        axes = axes.to(device=triplets.device, dtype=torch.long)
        if axes.numel() and (int(axes.min()) < 0 or int(axes.max()) > 2):
            raise ValueError("axes must contain only 0, 1, or 2.")

        forward = self.score_once(triplets, axes)
        if not self.reversal_invariant:
            return forward
        reverse = self.score_once(triplets.flip(1), axes)
        return CriticScores(
            logits_global=(forward.logits_global + reverse.logits_global) * 0.5,
            logits_local=(forward.logits_local + reverse.logits_local) * 0.5,
        )

    def score_once(
        self,
        triplets: torch.Tensor,
        axes: torch.Tensor,
    ) -> CriticScores:
        embedding = self.axis_mlp(self.axis_embedding(axes)).to(dtype=triplets.dtype)
        return self.score(triplets.flatten(1, 2), embedding)
