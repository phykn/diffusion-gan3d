from collections.abc import Sequence
from functools import cache, partial

import torch
import torch.nn.functional as F
from torch import nn

from src.model.conv import Conv3d
from src.model.layers import (
    INV_SQRT_THREE,
    INV_SQRT_TWO,
    NULL_DOMAIN,
    AdaptiveResBlock,
    SinusoidalEmbedding,
    embed_domain,
    embedding_mlp,
    run_block,
)
from src.prepare.profile import profile_field


def channel_norm_silu(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    h = x.float()
    var, mean = torch.var_mean(h, dim=1, keepdim=True, correction=0)
    h = ((h - mean) * torch.rsqrt(var + eps)).to(dtype)
    return F.silu(h * scale.to(dtype) + shift.to(dtype))


@cache
def fused_channel_norm_silu():
    # Eager MPS runs each step over the whole volume and keeps every
    # intermediate for backward; one compiled kernel makes the denoiser step
    # about an eighth faster and its peak memory about a third smaller.
    # Created on first use: importing the compiler takes about a second.
    return torch.compile(channel_norm_silu)


class ChannelNormSiLU3D(nn.Module):
    """Normalize each voxel over channels, scale and shift, then apply SiLU."""

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

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """scale and shift broadcast over x; they default to the learned affine."""
        if scale is None:
            shape = (1, -1, 1, 1, 1)
            scale, shift = self.scale.view(shape), self.shift.view(shape)
        norm_silu = (
            fused_channel_norm_silu() if x.device.type == "mps" else channel_norm_silu
        )
        return norm_silu(x, scale, shift, self.eps)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        # Convolving before enlarging costs an eighth of convolving after.
        return F.interpolate(self.conv(x), size=size, mode="nearest")


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
        time_scale: float = 1.0,
        height_enabled: bool = False,
        profile_enabled: bool = False,
        coarse_enabled: bool = False,
    ) -> None:
        super().__init__()
        multipliers = tuple(channel_multipliers)

        channels = tuple(base_channels * scale for scale in multipliers)
        if not channels or any(channel < 2 for channel in channels):
            raise ValueError(
                "denoiser channels must be at least two for channel normalization."
            )
        levels = len(channels)
        self.downsample_factor = 2 ** (levels - 1)
        self.gradient_checkpointing = gradient_checkpointing
        self.coarse_enabled = coarse_enabled

        self.time_emb = SinusoidalEmbedding(embedding_channels)
        self.time_scale = time_scale
        self.time_mlp = embedding_mlp(embedding_channels, embedding_channels)
        self.latent_mlp = embedding_mlp(latent_channels, embedding_channels)
        self.num_domains = num_domains
        self.domain_embedding = nn.Embedding(num_domains, embedding_channels)
        self.input = Conv3d(num_phases, channels[0], 3, padding=1)
        self.profile_input = (
            Conv3d(num_phases, channels[0], 3, padding=1, bias=False)
            if profile_enabled
            else None
        )
        self.height_input = (
            Conv3d(1, channels[0], 3, padding=1, bias=False) if height_enabled else None
        )

        block = partial(AdaptiveResBlock, ChannelNormSiLU3D, Conv3d)
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for idx, ch in enumerate(channels):
            self.encoder.append(block(ch, ch, embedding_channels))
            if idx + 1 < len(channels):
                self.downsample.append(
                    nn.Sequential(
                        Conv3d(ch, ch, 3, stride=2, padding=1),
                        Conv3d(ch, channels[idx + 1], 1),
                    )
                )

        self.middle = nn.ModuleList(
            block(channels[-1], channels[-1], embedding_channels) for _ in range(2)
        )

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.upsample.append(Upsample3D(ch, skip_ch))
            self.decoder.append(block(skip_ch, skip_ch, embedding_channels))
            ch = skip_ch

        self.output_norm = ChannelNormSiLU3D(channels[0])
        self.output = Conv3d(channels[0], num_phases, 3, padding=1)

        self.vf_mlp = None
        if not coarse_enabled:
            vf_out = nn.Linear(embedding_channels, embedding_channels)
            self.vf_mlp = nn.Sequential(
                nn.Linear(num_phases, embedding_channels), nn.SiLU(), vf_out
            )
            nn.init.zeros_(vf_out.weight)
            nn.init.zeros_(vf_out.bias)

        self.coarse_input = (
            Conv3d(num_phases, channels[0], 3, padding=1, bias=False)
            if coarse_enabled
            else None
        )
        self.corruption_embedding = (
            embedding_mlp(1, embedding_channels) if coarse_enabled else None
        )

    def forward(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        **conditions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict clean phase values in [-1, 1]; see compute_logits for conditions."""
        return self.decode(
            self.compute_logits(x_current, time, latent, domain, **conditions)
        )

    def compute_logits(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        vf_present: torch.Tensor | None = None,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
        profile_present: torch.Tensor | None = None,
        coarse: torch.Tensor | None = None,
        corruption_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.coarse_enabled:
            if any(
                value is not None
                for value in (vf, vf_present, profile, profile_present)
            ):
                raise ValueError("SR denoisers do not accept volume fractions.")
            if coarse is None or coarse.shape != x_current.shape:
                raise ValueError(
                    "SR requires coarse phase fractions on the output grid."
                )
        elif coarse is not None or corruption_level is not None:
            raise ValueError("coarse conditioning requires an SR denoiser.")
        emb = self.embed(x_current, time, latent, domain, vf, vf_present)
        x = self.input(x_current)
        if self.coarse_enabled:
            if corruption_level is None:
                corruption_level = x_current.new_zeros(x_current.shape[0])
            emb = emb + self.corruption_embedding(
                corruption_level.to(x_current).reshape(-1, 1)
            )
            x = x + self.coarse_input(coarse.to(x_current).mul(2).sub(1))
        if self.height_input is not None:
            if height is None:
                raise ValueError("height-conditioned denoiser requires a height field.")
            x = x + self.height_input(height.to(x_current))
        if profile is not None:
            if self.profile_input is None:
                raise ValueError("profile conditioning is disabled.")
            field = profile_field(profile.to(x_current), x_current.shape[-3:]) * 2 - 1
            if profile_present is not None:
                field = field * profile_present.to(field).reshape(-1, 1, 1, 1, 1)
            x = x + self.profile_input(field)
        checkpointing = self.gradient_checkpointing and self.training
        skips = []
        for idx, block in enumerate(self.encoder):
            x = run_block(block, x, emb, checkpointing)
            if idx < len(self.downsample):
                skips.append(x)
                x = self.downsample[idx](x)

        for block in self.middle:
            x = run_block(block, x, emb, checkpointing)

        for upsample, block, skip in zip(
            self.upsample,
            self.decoder,
            reversed(skips),
            strict=True,
        ):
            # Adding the skip instead of concatenating it keeps the full-resolution
            # decoder at its own width, which dominates the step time.
            x = (upsample(x, size=skip.shape[-3:]) + skip) * INV_SQRT_TWO
            x = run_block(block, x, emb, checkpointing)

        return self.output(self.output_norm(x))

    def apply_guidance_logits(
        self,
        x_current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        guidance: float,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        vf_present: torch.Tensor | None = None,
        height: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
        profile_present: torch.Tensor | None = None,
        coarse: torch.Tensor | None = None,
        corruption_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditions = dict(
            vf=vf,
            vf_present=vf_present,
            height=height,
            profile=profile,
            profile_present=profile_present,
            coarse=coarse,
            corruption_level=corruption_level,
        )
        if self.coarse_enabled:
            # SR CFG drops domain, retaining coarse material and height.
            unconditional_domain = torch.full_like(domain, NULL_DOMAIN)
            unconditional_conditions = conditions
        else:
            if coarse is not None or corruption_level is not None:
                raise ValueError("coarse conditioning requires an SR denoiser.")
            if vf is None and profile is None:
                return self.compute_logits(
                    x_current, time, latent, domain, height=height
                )
            # LR CFG drops measured conditions, retaining domain and height.
            unconditional_domain = domain
            unconditional_conditions = dict(height=height)

        if guidance == 1.0:
            return self.compute_logits(x_current, time, latent, domain, **conditions)
        unconditional = self.compute_logits(
            x_current,
            time,
            latent,
            unconditional_domain,
            **unconditional_conditions,
        )
        if guidance == 0.0:
            return unconditional
        conditional = self.compute_logits(x_current, time, latent, domain, **conditions)
        baseline = unconditional.float()
        return baseline + guidance * (conditional.float() - baseline)

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
        time_emb = self.time_mlp(
            self.time_emb(time * self.time_scale).to(dtype=inputs.dtype)
        )
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
