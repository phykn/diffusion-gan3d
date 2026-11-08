import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.config.io import GenerationSettings, load_generation_settings
from src.data.crops import phase_channels, resize_anchor
from src.device import check_device, default_device
from src.export import (
    build_diffusion,
    check_lr_source,
    export_config,
    load_denoiser,
    read_export,
)
from src.sampling.anchor import PlaneAnchor, TileAnchors
from src.sampling.conditions import LRConditions, SRConditions
from src.sampling.memory import MemoryEstimate, check_memory, estimate_memory
from src.sampling.plan import (
    TilePlan,
    check_non_negative,
    make_plan,
    parse_shape,
    sr_margin,
)
from src.sampling.sampler import Predictor, TileSampler

__all__ = ["LowResolutionAPI", "PlaneAnchor", "SuperResolutionAPI"]


def output_shape(blocks, shape, size, patch: int, overlap: int) -> tuple[int, ...]:
    """The requested output size; blocks, shape and size are exclusive."""
    given = [value is not None for value in (blocks, shape, size)]
    if sum(given) > 1:
        raise ValueError("give only one of blocks, shape and size.")
    if blocks is not None:
        counts = parse_shape(blocks, "blocks")
        if not 2 * overlap < patch:
            raise ValueError(f"blocks need 0 <= 2 * overlap < patch ({patch}).")
        result = tuple(patch + (n - 1) * (patch - 2 * overlap) for n in counts)
    elif shape is not None:
        result = parse_shape(shape, "shape")
    elif size is not None:
        result = parse_shape(size, "size")
    else:
        result = (patch,) * 3
    if any(value < patch for value in result):
        raise ValueError(f"every output axis must be at least the patch ({patch}).")
    return result


class LowResolutionAPI:
    def __init__(self, weights: str | Path | Mapping, device=None):
        self.device = check_device(default_device() if device is None else device)
        payload = read_export(weights, "lr")
        self.config = export_config(payload)
        model = load_denoiser(payload, self.config).to(self.device)
        diffusion = build_diffusion(self.config).to(self.device)
        data = self.config.data
        self.crop_size = data.crop_size
        self.input_size = data.lo_res_size
        self.num_phases = data.num_phases
        self.default_margin = model.downsample_factor
        self.settings: GenerationSettings = load_generation_settings()
        self.predictor = Predictor.from_config(
            model, diffusion, self.config, self.device
        )

    def prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape != (self.crop_size, self.crop_size):
            raise ValueError(
                f"image must be a {self.crop_size} x {self.crop_size} original crop."
            )
        return resize_anchor(image, self.input_size, self.num_phases)

    def plan(self, *, blocks=None, shape=None, size=None, overlap=None, margin=None):
        """The tile plan of a request. margin (default: the denoiser's
        downsample factor) is generated around the output and cropped off."""
        overlap = check_non_negative(
            self.settings.overlap if overlap is None else overlap, "overlap"
        )
        margin = self.default_margin if margin is None else margin
        patch = self.input_size
        return make_plan(
            output_shape(blocks, shape, size, patch, overlap), patch, margin, overlap
        )

    def estimate_memory(
        self,
        *,
        blocks=None,
        shape=None,
        size=None,
        overlap=None,
        margin=None,
        anchors=(),
        guidance=None,
        probabilities=False,
    ) -> MemoryEstimate:
        """Memory of generate with these options.

        Anchors count at any strength, and guidance other than 0 and 1 counts
        as CFG, since vf may be given.
        """
        plan = self.plan(
            blocks=blocks, shape=shape, size=size, overlap=overlap, margin=margin
        )
        guidance = self.settings.guidance if guidance is None else float(guidance)
        return self._estimate(
            plan, bool(len(tuple(anchors))), probabilities, guidance not in (0.0, 1.0)
        )

    def _estimate(self, plan: TilePlan, guided: bool, probabilities: bool, cfg: bool):
        return estimate_memory(
            plan,
            self.num_phases,
            self.config.model.generator.channels[0],
            self.device,
            probabilities=probabilities,
            guided=guided,
            profile=self.config.profile_enabled,
            cfg=cfg,
        )

    def check_memory(self, estimate: MemoryEstimate, storage: str = "auto") -> str:
        return check_memory(estimate, self.device, storage)

    def generate(self, anchors=(), **options) -> torch.Tensor:
        """Labels uint8 (D, H, W) on the CPU.

        guidance only acts when vf or a depth profile conditions the model;
        at guidance 1, or without either, each step is one forward pass.
        """
        return self._generate(anchors, probabilities=False, **options)

    def generate_probs(self, anchors=(), **options) -> torch.Tensor:
        """Phase fractions float32 (C, D, H, W) on the CPU; options as generate."""
        return self._generate(anchors, probabilities=True, **options)

    def _generate(
        self,
        anchors: Sequence = (),
        *,
        probabilities: bool,
        blocks=None,
        shape=None,
        size=None,
        vf=None,
        seed: int | None = None,
        guidance: float | None = None,
        anchor_strength: float | None = None,
        overlap: int | None = None,
        margin: int | None = None,
        storage: str = "auto",
        progress: bool = False,
        height_origin: float = 0.0,
        height_extent: float | None = None,
        vf_profile: Mapping | None = None,
    ) -> torch.Tensor:
        guidance = self.settings.guidance if guidance is None else float(guidance)
        strength = (
            self.settings.anchor_strength
            if anchor_strength is None
            else anchor_strength
        )
        plan = self.plan(
            blocks=blocks, shape=shape, size=size, overlap=overlap, margin=margin
        )
        conditions = LRConditions(
            self.config,
            plan,
            self.device,
            vf=vf,
            vf_profile=vf_profile,
            height_origin=height_origin,
            height_extent=height_extent,
        )
        anchors = TileAnchors(anchors, plan, self.num_phases, strength, self.device)
        cfg = conditions.guided and guidance not in (0.0, 1.0)
        estimate = self._estimate(plan, anchors.active, probabilities, cfg)
        storage = self.check_memory(estimate, storage)
        return TileSampler(self.predictor).sample(
            plan,
            seed=seed,
            conditions=conditions,
            guidance=guidance,
            guide=anchors if anchors.active else None,
            probabilities=probabilities,
            storage=storage,
            progress=progress,
        )


def lr_fractions(low: torch.Tensor, num_phases: int) -> torch.Tensor:
    """LR labels (D, H, W) or fractions (C, D, H, W) as float32 fractions."""
    if not isinstance(low, torch.Tensor):
        raise TypeError("low must be a tensor.")
    if low.dtype.is_floating_point:
        if low.ndim != 4 or low.shape[0] != num_phases:
            raise ValueError(f"LR fractions must have shape ({num_phases}, D, H, W).")
        low = low.detach().to("cpu", torch.float32)
        if not bool(torch.isfinite(low).all() & (low >= 0).all() & (low <= 1).all()):
            raise ValueError("LR fractions must be finite and in [0, 1].")
        return low
    if low.ndim != 3:
        raise ValueError("LR labels must have shape (D, H, W).")
    return phase_channels(low.detach().cpu()[None], num_phases)[0]


class SuperResolutionAPI:
    """HR volumes from LR ones; the SR export carries the LR model it refines.

    tile_size, overlap and margin are HR voxels. Without tile_size the whole
    volume is one tile.
    """

    def __init__(
        self,
        weights: str | Path | Mapping,
        device=None,
        lr_weights: str | Path | Mapping | None = None,
    ):
        self.device = check_device(default_device() if device is None else device)
        payload = read_export(weights, "sr")
        self.config = export_config(payload)
        model = load_denoiser(payload, self.config).to(self.device)
        diffusion = build_diffusion(self.config).to(self.device)
        data = self.config.data
        self.crop_size = data.crop_size
        self.lo_res_size = data.lo_res_size
        self.hi_res_size = data.hi_res_size
        self.scale_factor = self.config.scale_factor
        self.num_phases = data.num_phases
        self.default_margin = sr_margin(self.scale_factor)
        self.predictor = Predictor.from_config(
            model, diffusion, self.config, self.device
        )
        self.lr = LowResolutionAPI(
            payload["lr"] if lr_weights is None else lr_weights, self.device
        )
        check_lr_source(self.lr.config, self.config)

    def default_tiling(self, tile_size: int | None = None) -> tuple[int, int]:
        """HR tile edge (default: the training size) and its overlap, aligned to
        the scale and at most 8."""
        scale = self.scale_factor
        tile_size = self.hi_res_size if tile_size is None else tile_size
        if type(tile_size) is not int or tile_size < scale or tile_size % scale:
            raise ValueError(f"tile_size must be a positive multiple of {scale}.")
        return tile_size, min(8 // scale, (tile_size - 1) // (2 * scale)) * scale

    def plan(self, low_shape, *, tile_size=None, overlap=None, margin=None):
        """The HR tile plan for an LR volume of low_shape."""
        scale = self.scale_factor
        shape = tuple(n * scale for n in parse_shape(low_shape, "low shape"))
        margin = check_non_negative(
            self.default_margin if margin is None else margin, "margin"
        )
        if margin % scale:
            raise ValueError(f"margin must be a multiple of the LR/HR scale ({scale}).")
        if tile_size is None:
            overlap = check_non_negative(0 if overlap is None else overlap, "overlap")
            return make_plan(shape, shape, margin, overlap)
        tile_size, default_overlap = self.default_tiling(tile_size)
        overlap = check_non_negative(
            default_overlap if overlap is None else overlap, "overlap"
        )
        if 2 * overlap % scale:
            raise ValueError(f"2 * overlap must be a multiple of the scale ({scale}).")
        return make_plan(shape, tile_size, margin, overlap)

    def estimate_memory(
        self,
        low_shape,
        *,
        tile_size=None,
        overlap=None,
        margin=None,
        probabilities=False,
    ) -> MemoryEstimate:
        """Memory of refine for a CPU LR volume of low_shape."""
        plan = self.plan(low_shape, tile_size=tile_size, overlap=overlap, margin=margin)
        return self._estimate(plan, probabilities)

    def _estimate(self, plan: TilePlan, probabilities: bool) -> MemoryEstimate:
        scale = self.scale_factor
        low = math.prod(n // scale for n in plan.output_shape)
        padded = math.prod(n // scale for n in plan.shape)
        return estimate_memory(
            plan,
            self.num_phases,
            self.config.model.generator.channels[0],
            self.device,
            probabilities=probabilities,
            coarse=True,
            # The LR fractions and their padded copy stay in RAM.
            extra_host_bytes=4 * self.num_phases * (low + padded),
        )

    def check_memory(self, estimate: MemoryEstimate, storage: str = "auto") -> str:
        return check_memory(estimate, self.device, storage)

    def refine(self, low: torch.Tensor, **options) -> torch.Tensor:
        """HR labels uint8 (D, H, W) * scale on the CPU."""
        return self._refine(low, probabilities=False, **options)

    def refine_probs(self, low: torch.Tensor, **options) -> torch.Tensor:
        """HR phase fractions float32 (C, D, H, W) * scale; options as refine."""
        return self._refine(low, probabilities=True, **options)

    def _refine(
        self,
        low: torch.Tensor,
        *,
        probabilities: bool,
        seed: int | None = None,
        tile_size: int | None = None,
        overlap: int | None = None,
        margin: int | None = None,
        storage: str = "auto",
        height_origin: float = 0.0,
        height_extent: float | None = None,
    ) -> torch.Tensor:
        probs = lr_fractions(low, self.num_phases)
        plan = self.plan(
            probs.shape[1:], tile_size=tile_size, overlap=overlap, margin=margin
        )
        conditions = SRConditions(
            self.config,
            plan,
            probs,
            self.device,
            height_origin=height_origin,
            height_extent=height_extent,
        )
        storage = self.check_memory(self._estimate(plan, probabilities), storage)
        return TileSampler(self.predictor).sample(
            plan,
            seed=seed,
            conditions=conditions,
            probabilities=probabilities,
            storage=storage,
        )
