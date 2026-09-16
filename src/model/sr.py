import math

import torch
import torch.nn.functional as F
from torch import nn

from src.prepare.resize import scaled_size


class Residual3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.layers(x)) / math.sqrt(2)


class SuperResolution(nn.Module):
    def __init__(
        self,
        num_phases: int,
        scale_factor: float,
        num_domains: int = 1,
        channels: int = 16,
        blocks: int = 3,
        noise_channels: int = 1,
        height_enabled: bool = False,
    ):
        super().__init__()
        for name, value in (
            ("num_phases", num_phases),
            ("num_domains", num_domains),
            ("channels", channels),
            ("blocks", blocks),
            ("noise_channels", noise_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if num_phases < 2 or num_phases > 256:
            raise ValueError("num_phases must be between 2 and 256.")
        self.num_phases = num_phases
        self.scale_factor = scale_factor
        self.noise_channels = noise_channels
        self.height_input = (
            nn.Conv3d(1, channels, 3, padding=1, bias=False) if height_enabled else None
        )
        self.domain = nn.Embedding(num_domains, channels)
        self.corruption_embedding = nn.Sequential(
            nn.Linear(1, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.input = nn.Conv3d(num_phases + noise_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(Residual3D(channels) for _ in range(blocks)))
        self.output = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, num_phases, 3, padding=1),
        )

    def forward(
        self,
        low: torch.Tensor,
        noise: torch.Tensor,
        domain: torch.Tensor,
        corruption_level: torch.Tensor | None = None,
        height: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # low: B,K,D,H,W probabilities; spatial noise keeps each realization reproducible.
        shape = tuple(
            scaled_size(int(size), self.scale_factor) for size in low.shape[2:]
        )
        features = self.input(torch.cat((low, noise), dim=1))
        if self.height_input is not None:
            if height is None:
                raise ValueError("height-conditioned SR requires a height field.")
            features = features + self.height_input(height.to(low))
        features = features + self.domain(domain)[:, :, None, None, None]
        if corruption_level is None:
            corruption_level = low.new_zeros(low.shape[0])
        features = (
            features
            + self.corruption_embedding(corruption_level.to(low).reshape(-1, 1))[
                :, :, None, None, None
            ]
        )
        features = self.blocks(F.silu(features))
        features = F.interpolate(
            features, size=shape, mode="trilinear", align_corners=False
        )
        base = F.interpolate(low, size=shape, mode="trilinear", align_corners=False)
        return self.output(features) + base.clamp_min(0.01).log()
