import torch
import torch.nn.functional as F
from torch import nn

from src.model.pyramid import area_pyramid


class SliceCritic(nn.Module):
    def __init__(
        self,
        num_phases: int,
        channels: list[int],
        pyramid_min_size: int = 16,
        height_enabled: bool = False,
    ):
        super().__init__()
        if not channels or any(
            isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in channels
        ):
            raise ValueError("critic.channels must contain positive integers.")
        layers = []
        incoming = num_phases
        for width in channels:
            layers.extend(
                (nn.Conv2d(incoming, width, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
            )
            incoming = width
        layers.append(nn.Conv2d(incoming, 1, 1))
        self.layers = nn.Sequential(*layers)
        self.pyramid_min_size = pyramid_min_size
        self.height_input = (
            nn.Conv2d(1, num_phases, 1, bias=False) if height_enabled else None
        )

    def forward(self, images: torch.Tensor, height=None) -> torch.Tensor:
        return torch.stack(self.level_scores(images, height)).mean(0)

    def level_scores(self, images, height=None):
        scores = []
        for level in area_pyramid(images, self.pyramid_min_size):
            if height is not None and self.height_input is not None:
                level = level + self.height_input(
                    F.interpolate(height.to(level), size=level.shape[-2:], mode="area")
                )
            scores.append(self.layers(level).mean(dim=(1, 2, 3)))
        return tuple(scores)
