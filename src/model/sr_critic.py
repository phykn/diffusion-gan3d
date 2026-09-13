import torch
from torch import nn


class SliceCritic(nn.Module):
    def __init__(self, num_phases: int, channels: list[int]):
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).mean(dim=(1, 2, 3))
