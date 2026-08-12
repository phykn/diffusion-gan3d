import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    INV_SQRT_TWO,
    AdaptiveResBlock3D,
    ChannelNorm3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
)

MAX_GUIDANCE = 10_000.0


def validate_guidance(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0.0
        or value > MAX_GUIDANCE
    ):
        raise ValueError(
            f"guidance must be a finite number between zero and {MAX_GUIDANCE:g}."
        )
    return float(value)


class Denoiser3D(nn.Module):
    def __init__(
        self,
        num_phases: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        embedding_channels: int,
        latent_channels: int,
        num_domains: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        multipliers = tuple(channel_multipliers)

        channels = tuple(base_channels * scale for scale in multipliers)
        levels = len(channels)
        self.downsample_factor = 2 ** (levels - 1)
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
        self.num_domains = num_domains
        self.domain_embedding = nn.Embedding(num_domains, embedding_channels)
        self.input = nn.Conv3d(num_phases, channels[0], 3, padding=1)
        self.anchor_input = nn.Conv3d(
            num_phases + 1,
            channels[0],
            3,
            padding=1,
            bias=False,
        )
        # Zero initialization preserves unconditioned behavior at training start.
        nn.init.zeros_(self.anchor_input.weight)

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

        self.output_norm = ChannelNorm3D(channels[0])
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
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        vf_present: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(
            x_current,
            time,
            latent,
            domain,
            vf=vf,
            vf_present=vf_present,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        return self.decode(logits)

    def predict_logits(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        vf_present: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.embed(x_current, time, latent, domain, vf, vf_present)
        x = self.input(x_current)
        anchor_feat = self.encode_anchor(
            x_current,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        if anchor_feat is not None:
            x = x + anchor_feat
        skips = []
        for idx, block in enumerate(self.encoder):
            x = self.run_block(block, x, emb)
            if idx < len(self.downsample):
                skips.append(x)
                x = self.downsample[idx](x)

        for block in self.middle:
            x = self.run_block(block, x, emb)

        for upsample, block, skip in zip(
            self.upsample,
            self.decoder,
            reversed(skips),
            strict=True,
        ):
            x = upsample(x, size=skip.shape[-3:])
            x = torch.cat(
                (x * INV_SQRT_TWO, skip * INV_SQRT_TWO),
                dim=1,
            )
            x = self.run_block(block, x, emb)

        logits = self.output(F.silu(self.output_norm(x)))
        return logits

    def predict_guided(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        guidance: float,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        vf_present: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine conditional and unconditional logits with shared stochastic inputs."""
        guidance = validate_guidance(guidance)
        self._validate_vf_condition(x_current, vf, vf_present)
        if guidance == 1.0 or (
            vf is None and anchor_image is None and anchor_mask is None
        ):
            return self(
                x_current,
                time,
                latent,
                domain,
                vf=vf,
                vf_present=vf_present,
                anchor_image=anchor_image,
                anchor_mask=anchor_mask,
            )
        unconditional = self.predict_logits(x_current, time, latent, domain)
        if guidance == 0.0:
            return self.decode(unconditional)
        conditional = self.predict_logits(
            x_current,
            time,
            latent,
            domain,
            vf=vf,
            vf_present=vf_present,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        baseline = unconditional.to(torch.float32)
        guided = conditional.to(torch.float32)
        guided.sub_(baseline).mul_(guidance).add_(baseline)
        return self.decode(guided).to(x_current.dtype)

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return 2.0 * logits.softmax(dim=1) - 1.0

    def embed(
        self,
        inputs: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        vf: torch.Tensor | None,
        vf_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_vf_condition(inputs, vf, vf_present)
        time = time.to(device=inputs.device)
        latent = latent.to(device=inputs.device, dtype=inputs.dtype)
        time_emb = self.time_mlp(self.time_emb(time).to(dtype=inputs.dtype))
        latent_emb = self.latent_mlp(latent)
        emb = (time_emb + latent_emb) * INV_SQRT_TWO
        domain_emb = self.domain_embedding(domain.to(inputs.device)).to(inputs.dtype)
        emb = (emb + domain_emb) * INV_SQRT_TWO
        if vf is not None:
            vf = vf.to(device=inputs.device, dtype=inputs.dtype)
            vf_emb = self.vf_mlp(vf)
            if vf_present is None:
                emb = emb + vf_emb
            else:
                vf_present = vf_present.to(device=inputs.device)
                emb = torch.where(vf_present[:, None], emb + vf_emb, emb)
        return emb

    def _validate_vf_condition(
        self,
        inputs: torch.Tensor,
        vf: torch.Tensor | None,
        vf_present: torch.Tensor | None,
    ) -> None:
        if vf is None:
            if vf_present is not None:
                raise ValueError("vf_present requires vf.")
            return
        if not isinstance(vf, torch.Tensor) or not vf.is_floating_point():
            raise TypeError("vf must be a floating-point tensor.")
        expected = (inputs.shape[0], self.vf_mlp[0].in_features)
        if vf.shape != expected:
            raise ValueError("vf must have shape [B, num_phases].")
        if vf_present is None:
            return
        if not isinstance(vf_present, torch.Tensor) or vf_present.dtype != torch.bool:
            raise TypeError("vf_present must be a boolean tensor.")
        if vf_present.shape != (inputs.shape[0],):
            raise ValueError("vf_present must have shape [B].")

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

    def encode_anchor(
        self,
        inputs: torch.Tensor,
        anchor_image: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if anchor_image is None and anchor_mask is None:
            return None
        if anchor_image is None or anchor_mask is None:
            raise ValueError("anchor_image and anchor_mask must be provided together.")
        if not isinstance(anchor_image, torch.Tensor) or not isinstance(
            anchor_mask,
            torch.Tensor,
        ):
            raise TypeError("anchor_image and anchor_mask must be tensors.")
        if anchor_image.shape != inputs.shape:
            raise ValueError("anchor_image must have the same shape as inputs.")
        expected_mask = (inputs.shape[0], 1, *inputs.shape[2:])
        if anchor_mask.shape != expected_mask:
            raise ValueError("anchor_mask must have shape [B, 1, D, H, W].")
        mask = anchor_mask.to(device=inputs.device, dtype=inputs.dtype)
        clean = anchor_image.to(device=inputs.device, dtype=inputs.dtype)
        probs = (clean + 1.0) * 0.5 * mask
        return self.anchor_input(torch.cat((probs, mask), dim=1))
