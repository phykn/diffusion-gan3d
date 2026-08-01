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


class PairCritic2D(nn.Module):
    def __init__(
        self,
        num_phases: int,
        channels: Sequence[int],
        embedding_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2:
            raise ValueError("channels must contain at least two levels.")

        self.gradient_checkpointing = gradient_checkpointing
        self.time_embedding = SinusoidalTimeEmbedding(embedding_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.input = nn.Conv2d(2 * num_phases, widths[0], 3, padding=1)
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

    def forward(
        self,
        x_previous: torch.Tensor,
        x_current: torch.Tensor,
        time: torch.Tensor,
    ) -> CriticScores:
        emb = self.time_mlp(
            self.time_embedding(time.to(device=x_previous.device)).to(
                dtype=x_previous.dtype
            )
        )
        x = self.input(torch.cat((x_previous, x_current), dim=1))
        for idx, block in enumerate(self.blocks):
            x = self._run_block(block, x, emb)
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

    def _run_block(
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
