import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AdaptiveResBlock3D,
    Downsample3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
    check_channels,
    choose_groups,
)

_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


class Denoiser3D(nn.Module):
    def __init__(
        self,
        *,
        num_phases: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        embedding_channels: int,
        latent_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(num_phases, int) or num_phases < 2:
            raise ValueError("num_phases must be an integer of at least 2.")
        if not isinstance(base_channels, int) or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer.")
        if not isinstance(embedding_channels, int) or embedding_channels < 4:
            raise ValueError("embedding_channels must be an integer of at least 4.")
        if not isinstance(latent_channels, int) or latent_channels <= 0:
            raise ValueError("latent_channels must be a positive integer.")
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")
        multipliers = check_channels("channel_multipliers", channel_multipliers)

        channels = tuple(base_channels * multiplier for multiplier in multipliers)
        self.gradient_checkpointing = gradient_checkpointing

        self.time_embedding = SinusoidalTimeEmbedding(embedding_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.input = nn.Conv3d(num_phases, channels[0], 3, padding=1)
        self.anchor_input = nn.Conv3d(
            num_phases + 1,
            channels[0],
            3,
            padding=1,
        )
        # Zero initialization preserves unconditioned behavior at training start.
        nn.init.zeros_(self.anchor_input.weight)
        nn.init.zeros_(self.anchor_input.bias)

        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.encoder.append(
                AdaptiveResBlock3D(channel, channel, embedding_channels)
            )
            if index + 1 < len(channels):
                self.downsample.append(
                    nn.Sequential(
                        Downsample3D(channel),
                        nn.Conv3d(channel, channels[index + 1], 1),
                    )
                )

        self.middle = nn.ModuleList(
            (
                AdaptiveResBlock3D(
                    channels[-1],
                    channels[-1],
                    embedding_channels,
                ),
                AdaptiveResBlock3D(
                    channels[-1],
                    channels[-1],
                    embedding_channels,
                ),
            )
        )

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.upsample.append(Upsample3D(current, skip_channels))
            self.decoder.append(
                AdaptiveResBlock3D(
                    2 * skip_channels,
                    skip_channels,
                    embedding_channels,
                )
            )
            current = skip_channels

        self.output_norm = nn.GroupNorm(choose_groups(channels[0]), channels[0])
        self.output = nn.Conv3d(channels[0], num_phases, 3, padding=1)

    def forward(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(
            x_current,
            time,
            latent,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        return self.decode(logits)

    def predict_logits(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self._embed(x_current, time, latent)
        hidden = self.input(x_current)
        anchor_features = self._encode_anchor(
            x_current,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        if anchor_features is not None:
            hidden = hidden + anchor_features
        skips = []
        for index, block in enumerate(self.encoder):
            hidden = self._run_block(block, hidden, embedding)
            if index < len(self.downsample):
                skips.append(hidden)
                hidden = self.downsample[index](hidden)

        for block in self.middle:
            hidden = self._run_block(block, hidden, embedding)

        for upsample, block, skip in zip(
            self.upsample,
            self.decoder,
            reversed(skips),
            strict=True,
        ):
            hidden = upsample(hidden, size=skip.shape[-3:])
            hidden = torch.cat(
                (hidden * _INV_SQRT_TWO, skip * _INV_SQRT_TWO),
                dim=1,
            )
            hidden = self._run_block(block, hidden, embedding)

        logits = self.output(F.silu(self.output_norm(hidden)))
        return logits

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return 2.0 * logits.softmax(dim=1) - 1.0

    def _embed(
        self,
        inputs: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        time = time.to(device=inputs.device)
        latent = latent.to(device=inputs.device, dtype=inputs.dtype)
        time_embedding = self.time_mlp(self.time_embedding(time).to(dtype=inputs.dtype))
        latent_embedding = self.latent_mlp(latent)
        return (time_embedding + latent_embedding) * _INV_SQRT_TWO

    def _run_block(
        self,
        block: nn.Module,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                block,
                inputs,
                embedding,
                use_reentrant=False,
            )
        return block(inputs, embedding)

    def _encode_anchor(
        self,
        inputs: torch.Tensor,
        *,
        anchor_image: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if anchor_image is None or anchor_mask is None:
            return None
        mask = anchor_mask.to(device=inputs.device, dtype=inputs.dtype)
        active = mask.flatten(start_dim=1).any(dim=1)
        if not bool(active.any().item()):
            return None
        clean = anchor_image.to(device=inputs.device, dtype=inputs.dtype)
        probabilities = (clean + 1.0) * 0.5 * mask
        features = self.anchor_input(torch.cat((probabilities, mask), dim=1))
        return features * active.to(features.dtype)[:, None, None, None, None]
