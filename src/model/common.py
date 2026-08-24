import math

import torch
import torch.nn.functional as F
from torch import nn

INV_SQRT_TWO = 1.0 / math.sqrt(2.0)
INV_SQRT_THREE = 1.0 / math.sqrt(3.0)
NULL_DOMAIN = -1


def embed_domain(
    embedding: nn.Embedding,
    domain: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    domain = domain.to(device=embedding.weight.device, dtype=torch.long)
    if domain.numel() and (
        int(domain.min()) < NULL_DOMAIN or int(domain.max()) >= embedding.num_embeddings
    ):
        raise ValueError("domain contains an invalid ID.")
    present = domain != NULL_DOMAIN
    values = embedding(domain.clamp_min(0)).to(dtype=dtype)
    return values * present[:, None].to(dtype=dtype)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, channels: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        half = channels // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / denom
        )
        self.channels = channels
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError("value must have shape [B].")
        angles = value.to(dtype=torch.float32)[:, None] * self.freqs[None]
        emb = torch.cat((angles.cos(), angles.sin()), dim=1)
        if emb.shape[1] < self.channels:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaptiveNorm(nn.Module):
    def __init__(
        self,
        norm: type[nn.Module],
        channels: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.norm = norm(channels, affine=False)
        self.affine = nn.Linear(embedding_channels, 2 * channels)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.affine(F.silu(emb)).chunk(2, dim=1)
        dims = (1,) * (x.ndim - 2)
        scale = scale.reshape(scale.shape + dims)
        shift = shift.reshape(shift.shape + dims)
        return self.norm(x) * (1.0 + scale) + shift
