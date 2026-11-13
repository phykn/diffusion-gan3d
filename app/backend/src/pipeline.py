from dataclasses import dataclass

from src.api import SuperResolutionAPI
from src.sampling.memory import MemoryEstimate


@dataclass(frozen=True)
class PipelineEstimate:
    low: MemoryEstimate
    high: MemoryEstimate

    @property
    def shape(self):
        return self.high.shape

    @property
    def tile_count(self):
        return self.low.tiles


class HighResolutionPipeline:
    def __init__(self, weights, lr_weights=None, device=None):
        self.sr = SuperResolutionAPI(weights, device, lr_weights=lr_weights)
        self.lr = self.sr.lr
        self.sr_tiling = dict(
            zip(("tile_size", "overlap"), self.sr.default_tiling(), strict=True)
        )

    @property
    def device(self):
        return self.lr.device

    @property
    def crop_size(self):
        return self.lr.crop_size

    @property
    def num_phases(self):
        return self.lr.num_phases

    @property
    def height_enabled(self):
        return self.lr.config.conditioning.height_enabled

    @property
    def height_extent(self):
        return self.lr.config.data.height_extent if self.height_enabled else None

    def prepare_image(self, image):
        return self.lr.prepare_image(image)

    def estimate_memory(self, *, blocks=None, shape=None, anchors=()):
        low = self.lr.estimate_memory(
            blocks=blocks, shape=shape, anchors=anchors, probabilities=True
        )
        return PipelineEstimate(
            low, self.sr.estimate_memory(low.shape, **self.sr_tiling)
        )

    def check_memory(self, estimate):
        self.lr.check_memory(estimate.low)
        self.sr.check_memory(estimate.high)

    def generate(self, *, seed=None, height_origin=0.0, height_extent=None, **options):
        conditions = {
            "seed": seed,
            "height_origin": height_origin,
            "height_extent": height_extent,
        }
        coarse = self.lr.generate_probs(**conditions, **options)
        return self.sr.refine(coarse, **self.sr_tiling, **conditions)
