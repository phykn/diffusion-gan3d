from dataclasses import dataclass
from pathlib import Path

import torch

from src.api import LowResolutionAPI, SuperResolutionAPI
from src.config.data import validate_sr_source
from src.predict.memory import MemoryEstimate
from src.predict.sr.memory import SRMemoryEstimate
from src.train.run.source import frozen_source_weights


@dataclass(frozen=True)
class PipelineEstimate:
    low: MemoryEstimate
    high: SRMemoryEstimate

    @property
    def shape(self) -> tuple[int, int, int]:
        """The HR output shape."""
        return self.high.shape

    @property
    def tile_count(self) -> int:
        # Requests set the LR overlap; the SR tiling is fixed by the server.
        return self.low.tile_count


class HighResolutionPipeline:
    """Generate LR phase fractions from the request, then refine them to HR labels.

    The LR model is the one the SR run was trained from unless lr_weights is given.
    """

    def __init__(
        self,
        weights: str | Path,
        lr_weights: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.sr = SuperResolutionAPI(weights, device)
        if lr_weights is None:
            lr_weights = frozen_source_weights(self.sr.config.get("source", {}))
        self.lr = LowResolutionAPI(lr_weights, self.sr.device)
        validate_sr_source(self.sr.config["data"], self.lr.data)
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
    def generator(self):
        """The LR generator, whose domains and height inputs requests address."""
        return self.lr.generator

    def prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.lr.prepare_image(image)

    def estimate_memory(self, **options) -> PipelineEstimate:
        low = self.lr.estimate_memory(probabilities=True, **options)
        high = self.sr.estimate_memory(
            low.shape,
            tile_size=self.tile_size,
            overlap=self.overlap,
            label_input=False,
            input_element_size=4,
        )
        return PipelineEstimate(low, high)

    def check_memory(self, estimate: PipelineEstimate, *, storage="auto") -> None:
        self.lr.check_memory(estimate.low, storage=storage)
        self.sr.check_memory(estimate.high)

    def generate(
        self,
        *,
        domain=None,
        seed=None,
        height_origin=0.0,
        height_extent=None,
        **options,
    ) -> torch.Tensor:
        """HR uint8 labels; options are LowResolutionAPI.generate_probs's."""
        common = dict(
            domain=domain,
            seed=seed,
            height_origin=height_origin,
            height_extent=height_extent,
        )
        low = self.lr.generate_probs(**common, **options)
        return self.sr.refine(
            low, tile_size=self.tile_size, overlap=self.overlap, **common
        )
