from functools import cache, partial

import torch
import torch.nn.functional as F
from torch import nn

from src.model.conv import Conv3d
from src.model.layers import (
    INV_SQRT_TWO,
    AdaptiveResBlock,
    SinusoidalEmbedding,
    build_embedding_mlp,
    run_block,
)

MEASURED_CONDITIONS = ("vf", "vf_present", "profile", "profile_present")


class ChannelNormSiLU3D(nn.Module):
    def __init__(self, channels, eps=1.0e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("scale", None)
            self.register_parameter("shift", None)

    def forward(self, x, scale=None, shift=None):
        if scale is None:
            per_channel = (1, -1, 1, 1, 1)
            scale, shift = self.scale.view(per_channel), self.shift.view(per_channel)
        # One dtype and no stray requires_grad keep the MPS kernel from recompiling.
        scale, shift = scale.float(), shift.float()
        if not torch.is_grad_enabled():
            scale, shift = scale.detach(), shift.detach()
        norm_silu = compile_norm_silu() if x.device.type == "mps" else apply_norm_silu
        return norm_silu(x, scale, shift, self.eps)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, size):
        # Convolving before enlarging costs an eighth of convolving after.
        return F.interpolate(self.conv(x), size=size, mode="nearest")


class Denoiser3D(nn.Module):
    def __init__(
        self,
        num_phases,
        base_channels,
        channel_multipliers,
        embedding_channels,
        latent_channels,
        gradient_checkpointing=False,
        height_enabled=False,
        profile_enabled=False,
        coarse_enabled=False,
    ):
        super().__init__()
        widths = tuple(base_channels * multiplier for multiplier in channel_multipliers)
        if not widths or any(width < 2 for width in widths):
            raise ValueError(
                "denoiser channels must be at least two for channel normalization."
            )
        if coarse_enabled and profile_enabled:
            raise ValueError("SR denoisers do not accept a depth profile.")
        self.downsample_factor = 2 ** (len(widths) - 1)
        self.gradient_checkpointing = gradient_checkpointing
        self.coarse_enabled = coarse_enabled

        self.time_emb = SinusoidalEmbedding(embedding_channels)
        self.time_mlp = build_embedding_mlp(embedding_channels, embedding_channels)
        self.latent_mlp = build_embedding_mlp(latent_channels, embedding_channels)
        self.input = Conv3d(num_phases, widths[0], 3, padding=1)
        self.profile_input = (
            Conv3d(num_phases, widths[0], 3, padding=1, bias=False)
            if profile_enabled
            else None
        )
        self.height_input = (
            Conv3d(1, widths[0], 3, padding=1, bias=False) if height_enabled else None
        )

        make_block = partial(AdaptiveResBlock, ChannelNormSiLU3D, Conv3d)
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for level, width in enumerate(widths):
            self.encoder.append(make_block(width, embedding_channels))
            if level + 1 < len(widths):
                self.downsample.append(
                    nn.Sequential(
                        Conv3d(width, width, 3, stride=2, padding=1),
                        Conv3d(width, widths[level + 1], 1),
                    )
                )

        self.middle = nn.ModuleList(
            make_block(widths[-1], embedding_channels) for _ in range(2)
        )

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        width = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.upsample.append(Upsample3D(width, skip_width))
            self.decoder.append(make_block(skip_width, embedding_channels))
            width = skip_width

        self.output_norm = ChannelNormSiLU3D(widths[0])
        self.output = Conv3d(widths[0], num_phases, 3, padding=1)

        self.vf_mlp = None
        if not coarse_enabled:
            vf_last = nn.Linear(embedding_channels, embedding_channels)
            self.vf_mlp = nn.Sequential(
                nn.Linear(num_phases, embedding_channels), nn.SiLU(), vf_last
            )
            nn.init.zeros_(vf_last.weight)
            nn.init.zeros_(vf_last.bias)

        self.coarse_input = (
            Conv3d(num_phases, widths[0], 3, padding=1, bias=False)
            if coarse_enabled
            else None
        )
        self.corruption_embedding = (
            build_embedding_mlp(1, embedding_channels) if coarse_enabled else None
        )

    def forward(self, x_current, time, latent, **conditions):
        return self.decode(self.predict_logits(x_current, time, latent, **conditions))

    def predict_logits(
        self,
        x_current,
        time,
        latent,
        vf=None,
        vf_present=None,
        height=None,
        profile=None,
        profile_present=None,
        coarse=None,
        corruption_level=None,
    ):
        if self.coarse_enabled and vf_present is not None:
            raise ValueError("SR denoisers do not accept volume fractions.")
        embedding = self.embed(time, latent.to(x_current), vf, vf_present)
        if self.coarse_enabled:
            if profile is not None or profile_present is not None:
                raise ValueError("SR denoisers do not accept a depth profile.")
            if coarse is None or coarse.shape != x_current.shape:
                raise ValueError(
                    "SR requires coarse phase fractions on the output grid."
                )
        elif coarse is not None or corruption_level is not None:
            raise ValueError("coarse conditioning requires an SR denoiser.")
        x = self.input(x_current)
        if self.coarse_enabled:
            if corruption_level is None:
                corruption_level = x_current.new_zeros(x_current.shape[0])
            embedding = embedding + self.corruption_embedding(
                corruption_level.to(x_current).reshape(-1, 1)
            )
            x = x + self.coarse_input(coarse.to(x_current).mul(2).sub(1))
        if self.height_input is not None:
            if height is None:
                raise ValueError("height-conditioned denoiser requires a height field.")
            x = x + self.height_input(height.to(x_current))
        elif height is not None:
            raise ValueError("height conditioning is disabled.")
        if profile is not None:
            x = x + self._embed_profile(x_current, profile, profile_present)
        return self._run_unet(x, embedding)

    def predict_guided_logits(self, x_current, time, latent, guidance, **conditions):
        if (
            self.coarse_enabled
            or guidance == 1.0
            or (conditions.get("vf") is None and conditions.get("profile") is None)
        ):
            return self.predict_logits(x_current, time, latent, **conditions)
        unmeasured = {
            key: value
            for key, value in conditions.items()
            if key not in MEASURED_CONDITIONS
        }
        unconditional = self.predict_logits(x_current, time, latent, **unmeasured)
        if guidance == 0.0:
            return unconditional
        conditional = self.predict_logits(x_current, time, latent, **conditions)
        baseline = unconditional.float()
        return baseline + guidance * (conditional.float() - baseline)

    def embed(self, time, latent, vf=None, vf_present=None):
        time_embedding = self.time_mlp(
            self.time_emb(time.to(latent.device)).to(dtype=latent.dtype)
        )
        embedding = (time_embedding + self.latent_mlp(latent)) * INV_SQRT_TWO
        if vf is None:
            return embedding
        if self.vf_mlp is None:
            raise ValueError("SR denoisers do not accept volume fractions.")
        vf_embedding = self.vf_mlp(vf.to(device=latent.device, dtype=latent.dtype))
        if vf_present is None:
            return embedding + vf_embedding
        vf_present = vf_present.to(device=latent.device)
        return torch.where(vf_present[:, None], embedding + vf_embedding, embedding)

    @staticmethod
    def decode(logits):
        return 2.0 * logits.softmax(dim=1) - 1.0

    def _embed_profile(self, x_current, profile, profile_present):
        if self.profile_input is None:
            raise ValueError("profile conditioning is disabled.")
        shape = x_current.shape[-3:]
        if profile.ndim != 3 or profile.shape[-1] != shape[0]:
            raise ValueError("profile must match the volume depth.")
        layers = profile.to(x_current)[..., None, None].expand(-1, -1, *shape)
        layers = layers * 2 - 1
        if profile_present is not None:
            layers = layers * profile_present.to(layers).reshape(-1, 1, 1, 1, 1)
        return self.profile_input(layers)

    def _run_unet(self, x, embedding):
        checkpointing = self.gradient_checkpointing and self.training
        skips = []
        for level, block in enumerate(self.encoder):
            x = run_block(block, x, embedding, checkpointing)
            if level < len(self.downsample):
                skips.append(x)
                x = self.downsample[level](x)

        for block in self.middle:
            x = run_block(block, x, embedding, checkpointing)

        for upsample, block, skip in zip(
            self.upsample, self.decoder, reversed(skips), strict=True
        ):
            # Adding the skip keeps the decoder narrower than concatenating it.
            x = (upsample(x, size=skip.shape[-3:]) + skip) * INV_SQRT_TWO
            x = run_block(block, x, embedding, checkpointing)

        return self.output(self.output_norm(x))


def apply_norm_silu(x, scale, shift, eps):
    dtype = x.dtype
    normalized = x.float()
    variance, mean = torch.var_mean(normalized, dim=1, keepdim=True, correction=0)
    normalized = ((normalized - mean) * torch.rsqrt(variance + eps)).to(dtype)
    return F.silu(normalized * scale.to(dtype) + shift.to(dtype))


@cache
def compile_norm_silu():
    # On MPS one compiled kernel is faster and uses less memory than eager steps.
    return torch.compile(apply_norm_silu)
