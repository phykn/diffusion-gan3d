from dataclasses import dataclass
from pathlib import Path

import torch

from src.api import SuperResolutionAPI
from src.sampling.memory import MemoryEstimate


@dataclass(frozen=True)
class PipelineEstimate:
    low: MemoryEstimate
    high: MemoryEstimate

    @property
    def shape(self) -> tuple[int, int, int]:
        """The HR output shape."""
        return self.high.shape

    @property
    def tile_count(self) -> int:
        # Requests set the LR overlap; the SR tiling is fixed by the server.
        return self.low.tiles


class HighResolutionPipeline:
    """Generate LR phase fractions from the request, then refine them to HR labels.

    The LR model is the one stored in the SR export unless lr_weights is given.
    """

    def __init__(
        self,
        weights: str | Path,
        lr_weights: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.sr = SuperResolutionAPI(weights, device, lr_weights=lr_weights)
        self.lr = self.sr.lr
        self.tile_size, self.overlap = self.sr.default_tiling()

    @property
    def device(self) -> torch.device:
        return self.lr.device

    @property
    def crop_size(self) -> int:
        return self.lr.crop_size

    @property
    def num_phases(self) -> int:
        return self.lr.num_phases

    @property
    def height_enabled(self) -> bool:
        return self.lr.config.conditioning.height_enabled

    @property
    def height_extent(self) -> int | None:
        return self.lr.config.data.height_extent if self.height_enabled else None

    def prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.lr.prepare_image(image)

    def estimate_memory(
        self,
        *,
        blocks=None,
        shape=None,
        size=None,
        overlap=None,
        anchors=(),
        guidance=None,
    ) -> PipelineEstimate:
        """LR and SR memory of one request, with its anchors and guidance."""
        low = self.lr.estimate_memory(
            blocks=blocks,
            shape=shape,
            size=size,
            overlap=overlap,
            anchors=anchors,
            guidance=guidance,
            probabilities=True,
        )
        high = self.sr.estimate_memory(
            low.shape, tile_size=self.tile_size, overlap=self.overlap
        )
        return PipelineEstimate(low, high)

    def check_memory(self, estimate: PipelineEstimate, *, storage="auto") -> None:
        self.lr.check_memory(estimate.low, storage)
        self.sr.check_memory(estimate.high)

    def generate(
        self, *, seed=None, height_origin=0.0, height_extent=None, **options
    ) -> torch.Tensor:
        """HR uint8 labels; options are LowResolutionAPI.generate_probs's."""
        common = {
            "seed": seed,
            "height_origin": height_origin,
            "height_extent": height_extent,
        }
        low = self.lr.generate_probs(**common, **options)
        return self.sr.refine(
            low, tile_size=self.tile_size, overlap=self.overlap, **common
        )
