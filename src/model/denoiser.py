from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .common import (
    INV_SQRT_THREE,
    INV_SQRT_TWO,
    AdaptiveNorm,
    SinusoidalEmbedding,
    embed_domain,
)


class ChannelNorm3D(nn.Module):
    def __init__(
        self,
        channels: int,
        eps: float = 1.0e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("scale", None)
            self.register_parameter("shift", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(
            x,
            dim=1,
            keepdim=True,
            correction=0,
        )
        mean = mean.to(dtype=x.dtype)
        inv_std = torch.rsqrt(var + self.eps).to(dtype=x.dtype)
        x = (x - mean) * inv_std
        if self.scale is None:
            return x
        shape = (1, -1, 1, 1, 1)
        scale = self.scale.to(dtype=x.dtype).view(shape)
        shift = self.shift.to(dtype=x.dtype).view(shape)
        return x * scale + shift


class AdaptiveResBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = AdaptiveNorm(ChannelNorm3D, in_channels, embedding_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveNorm(ChannelNorm3D, out_channels, embedding_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, emb)))
        h = self.conv2(F.silu(self.norm2(h, emb)))
        return (self.skip(x) + h) * INV_SQRT_TWO


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        h = F.interpolate(x, size=size, mode="nearest")
        return self.conv(h)


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
        anchor_multiscale: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(anchor_multiscale, bool):
            raise TypeError("anchor_multiscale must be a boolean.")
        multipliers = tuple(channel_multipliers)

        channels = tuple(base_channels * scale for scale in multipliers)
        levels = len(channels)
        self.downsample_factor = 2 ** (levels - 1)
        self.gradient_checkpointing = gradient_checkpointing
        self.anchor_multiscale = anchor_multiscale

        self.time_emb = SinusoidalEmbedding(embedding_channels)
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

        # Build optional weights last so the shared backbone keeps identical RNG init.
        self.anchor_pyramid = nn.ModuleList()
        if anchor_multiscale:
            for channel in channels[1:]:
                projection = nn.Conv3d(
                    num_phases + 1,
                    channel,
                    1,
                    bias=False,
                )
                nn.init.zeros_(projection.weight)
                self.anchor_pyramid.append(projection)

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
        logits = self.compute_logits(
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

    def compute_logits(
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
        anchor = self._prepare_anchor(
            x_current,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        if anchor is not None:
            x = x + self.anchor_input(torch.cat(anchor, dim=1))
        skips = []
        for idx, block in enumerate(self.encoder):
            x = self.apply_block(block, x, emb)
            if idx < len(self.downsample):
                skips.append(x)
                x = self.downsample[idx](x)
                if anchor is not None and idx < len(self.anchor_pyramid):
                    anchor_at_scale = self._pool_anchor(
                        *anchor,
                        output_size=x.shape[-3:],
                    )
                    x = x + self.anchor_pyramid[idx](anchor_at_scale)

        for block in self.middle:
            x = self.apply_block(block, x, emb)

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
            x = self.apply_block(block, x, emb)

        logits = self.output(F.silu(self.output_norm(x)))
        return logits

    def apply_guidance_logits(
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
        if guidance == 0.0 or (
            vf is None and anchor_image is None and anchor_mask is None
        ):
            return self.compute_logits(x_current, time, latent, domain)
        elif guidance == 1.0:
            return self.compute_logits(
                x_current,
                time,
                latent,
                domain,
                vf=vf,
                vf_present=vf_present,
                anchor_image=anchor_image,
                anchor_mask=anchor_mask,
            )
        else:
            unconditional = self.compute_logits(x_current, time, latent, domain)
            conditional = self.compute_logits(
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
            conditional = conditional.to(torch.float32)
            return baseline + guidance * (conditional - baseline)

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
        time = time.to(device=inputs.device)
        latent = latent.to(device=inputs.device, dtype=inputs.dtype)
        time_emb = self.time_mlp(self.time_emb(time).to(dtype=inputs.dtype))
        latent_emb = self.latent_mlp(latent)
        domain_emb = embed_domain(
            self.domain_embedding,
            domain,
            inputs.dtype,
        )
        emb = (time_emb + latent_emb + domain_emb) * INV_SQRT_THREE
        if vf is not None:
            vf = vf.to(device=inputs.device, dtype=inputs.dtype)
            vf_emb = self.vf_mlp(vf)
            if vf_present is None:
                emb = emb + vf_emb
            else:
                vf_present = vf_present.to(device=inputs.device)
                emb = torch.where(vf_present[:, None], emb + vf_emb, emb)
        return emb

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

    def _prepare_anchor(
        self,
        inputs: torch.Tensor,
        anchor_image: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if anchor_image is None and anchor_mask is None:
            return None
        if anchor_image is None or anchor_mask is None:
            raise ValueError("anchor_image and anchor_mask must be provided together.")
        mask = anchor_mask.to(device=inputs.device, dtype=inputs.dtype)
        values = anchor_image.to(device=inputs.device, dtype=inputs.dtype)
        return values * mask, mask

    @staticmethod
    def _pool_anchor(
        values: torch.Tensor,
        mask: torch.Tensor,
        output_size: Sequence[int],
    ) -> torch.Tensor:
        coverage = F.adaptive_avg_pool3d(mask, output_size)
        pooled_values = F.adaptive_avg_pool3d(values, output_size)
        pooled_values = pooled_values / coverage.clamp_min(
            torch.finfo(values.dtype).eps,
        )
        return torch.cat((pooled_values, coverage), dim=1)
