import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AdaptiveResBlock3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
    choose_groups,
)

_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


class Denoiser3D(nn.Module):
    def __init__(
        self,
        num_phases: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        embedding_channels: int,
        latent_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        multipliers = tuple(channel_multipliers)

        channels = tuple(base_channels * scale for scale in multipliers)
        self.gradient_checkpointing = gradient_checkpointing

        self.time_emb = SinusoidalTimeEmbedding(embedding_channels)
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
        for idx, ch in enumerate(channels):
            self.encoder.append(AdaptiveResBlock3D(ch, ch, embedding_channels))
            if idx + 1 < len(channels):
                self.downsample.append(
                    nn.Sequential(
                        nn.Conv3d(ch, ch, 3, stride=2, padding=1),
                        nn.Conv3d(ch, channels[idx + 1], 1),
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
        ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.upsample.append(Upsample3D(ch, skip_ch))
            self.decoder.append(
                AdaptiveResBlock3D(
                    2 * skip_ch,
                    skip_ch,
                    embedding_channels,
                )
            )
            ch = skip_ch

        self.output_norm = nn.GroupNorm(choose_groups(channels[0]), channels[0])
        self.output = nn.Conv3d(channels[0], num_phases, 3, padding=1)

        vf_out = nn.Linear(embedding_channels, embedding_channels)
        self.vf_mlp = nn.Sequential(
            nn.Linear(num_phases, embedding_channels),
            nn.SiLU(),
            vf_out,
        )
        nn.init.zeros_(vf_out.weight)
        nn.init.zeros_(vf_out.bias)

    def forward(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(
            x_current,
            time,
            latent,
            vf=vf,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        return self.decode(logits)

    def predict_logits(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self._embed(x_current, time, latent, vf=vf)
        x = self.input(x_current)
        anchor_feat = self._encode_anchor(
            x_current,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        if anchor_feat is not None:
            x = x + anchor_feat
        skips = []
        for idx, block in enumerate(self.encoder):
            x = self._run_block(block, x, emb)
            if idx < len(self.downsample):
                skips.append(x)
                x = self.downsample[idx](x)

        for block in self.middle:
            x = self._run_block(block, x, emb)

        for upsample, block, skip in zip(
            self.upsample,
            self.decoder,
            reversed(skips),
            strict=True,
        ):
            x = upsample(x, size=skip.shape[-3:])
            x = torch.cat(
                (x * _INV_SQRT_TWO, skip * _INV_SQRT_TWO),
                dim=1,
            )
            x = self._run_block(block, x, emb)

        logits = self.output(F.silu(self.output_norm(x)))
        return logits

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return 2.0 * logits.softmax(dim=1) - 1.0

    def _embed(
        self,
        inputs: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        vf: torch.Tensor | None,
    ) -> torch.Tensor:
        time = time.to(device=inputs.device)
        latent = latent.to(device=inputs.device, dtype=inputs.dtype)
        time_emb = self.time_mlp(self.time_emb(time).to(dtype=inputs.dtype))
        latent_emb = self.latent_mlp(latent)
        emb = (time_emb + latent_emb) * _INV_SQRT_TWO
        if vf is not None:
            vf = vf.to(device=inputs.device, dtype=inputs.dtype)
            emb = emb + self.vf_mlp(vf)
        return emb

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

    def _encode_anchor(
        self,
        inputs: torch.Tensor,
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
        probs = (clean + 1.0) * 0.5 * mask
        feat = self.anchor_input(torch.cat((probs, mask), dim=1))
        return feat * active.to(feat.dtype)[:, None, None, None, None]
