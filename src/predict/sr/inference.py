import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from src.build.model import build_denoiser
from src.build.predict import make_generator, resolve_weights
from src.config.train import get_sr_resolution, normalize_train_config
from src.device import check_device, default_device
from src.predict.convert import labels_from_channels, owned_clean_to_probs_
from src.predict.sr.memory import (
    SRMemoryEstimate,
    check_sr_memory,
    estimate_sr_memory,
)
from src.predict.tiling.layout import (
    check_overlap,
    make_plan,
    make_tiles,
    parse_shape,
)
from src.predict.tiling.sampler import TileSampler
from src.predict.volume_condition import (
    fraction_channels,
    prepare_base,
    resolve_offset,
    volume_shape,
)
from src.prepare.resize import coarse_region, resize_phases, scaled_size
from src.random import seeded_rng
from src.storage import SR_FORMAT


def load_sr_export(path: str | Path) -> dict:
    """Load an exported SR generator.pt and return it with its resolved config."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != SR_FORMAT:
        raise ValueError(
            "use exported SR weights (generator.pt), not a training or LR checkpoint."
        )
    return {**payload, "config": normalize_train_config(payload["config"], "sr")}


class SuperResolutionAPI:
    """Refine an LR label/fraction volume into HR voxels using separate SR weights."""

    def __init__(
        self, weights: str | Path, device: str | torch.device | None = None
    ) -> None:
        device = check_device(default_device() if device is None else device)
        self.weights = resolve_weights(weights)
        payload = load_sr_export(self.weights)
        model = build_denoiser(payload["config"]).to(device)
        model.load_state_dict(payload["model"], strict=True)
        self._attach(model.eval().requires_grad_(False), payload["config"], device)

    @classmethod
    def from_model(cls, model, config: dict, device: torch.device):
        """Refine with a model already in memory, such as a training EMA."""
        api = cls.__new__(cls)
        api.weights = None
        api._attach(model, config, device)
        return api

    def _attach(self, model, config: dict, device: torch.device) -> None:
        self.device = device
        self.config = config
        self.model = model
        resolution = get_sr_resolution(self.config)
        self.crop_size = resolution.crop_size
        self.lo_res_size = resolution.lo_res_size
        self.hi_res_size = resolution.hi_res_size
        self.scale_factor = self.hi_res_size / self.lo_res_size
        self.num_phases = self.config["data"]["num_phases"]
        self.generator = make_generator(
            self.model, self.config, self.device, self.hi_res_size
        )

    @property
    def default_margin(self) -> int:
        """HR margin refine uses when none is given, whole LR cells at integer scales."""
        margin = self.generator.default_margin
        if self.scale_factor.is_integer():
            margin = math.lcm(margin, int(self.scale_factor))
        return margin

    def default_tiling(self, tile_size: int | None = None) -> tuple[int | None, int]:
        """HR tile edge (default: the training size) and its scale-aligned overlap
        up to 8.

        Only integer scales tile, so a fractional scale refines in one block.
        """
        integer = self.scale_factor.is_integer()
        scale = int(self.scale_factor) if integer else 1
        if tile_size is None and integer:
            tile_size = max(scale, self.hi_res_size // scale * scale)
        if tile_size is None:
            return None, 0
        return tile_size, min(8 // scale, (tile_size - 1) // (2 * scale)) * scale

    def estimate_memory(
        self,
        low_shape,
        *,
        tile_size: int | None = None,
        overlap: int | None = None,
        margin: int | None = None,
        label_input: bool = True,
        input_element_size: int = 8,
        probabilities: bool = False,
    ) -> SRMemoryEstimate:
        """Bytes to refine a CPU LR volume of low_shape; options are refine's."""
        low_shape = parse_shape(low_shape)
        shape, margin, _, _ = self._layout(low_shape, tile_size, overlap, margin)
        return estimate_sr_memory(
            low_shape,
            shape,
            self.num_phases,
            tile_size=tile_size,
            margin=margin,
            label_input=label_input,
            input_element_size=input_element_size,
            probabilities=probabilities,
        )

    def check_memory(self, estimate: SRMemoryEstimate) -> None:
        check_sr_memory(estimate, self.generator)

    def refine(
        self,
        low: torch.Tensor,
        *,
        domain: int | None = None,
        seed: int | None = None,
        guidance: float = 1.0,
        tile_size: int | None = None,
        overlap: int | None = None,
        margin: int | None = None,
        height_origin: float = 0.0,
        height_extent: float | None = None,
        base: torch.Tensor | None = None,
        base_offset=None,
    ) -> torch.Tensor:
        """Refine LR labels or phase fractions into HR uint8 labels."""
        return self._run(
            low,
            probabilities=False,
            domain=domain,
            seed=seed,
            guidance=guidance,
            tile_size=tile_size,
            overlap=overlap,
            margin=margin,
            height_origin=height_origin,
            height_extent=height_extent,
            base=base,
            base_offset=base_offset,
        )

    def refine_probs(
        self,
        low: torch.Tensor,
        *,
        domain: int | None = None,
        seed: int | None = None,
        guidance: float = 1.0,
        tile_size: int | None = None,
        overlap: int | None = None,
        margin: int | None = None,
        height_origin: float = 0.0,
        height_extent: float | None = None,
        base: torch.Tensor | None = None,
        base_offset=None,
    ) -> torch.Tensor:
        """Refine LR labels or phase fractions into HR phase fractions."""
        return self._run(
            low,
            probabilities=True,
            domain=domain,
            seed=seed,
            guidance=guidance,
            tile_size=tile_size,
            overlap=overlap,
            margin=margin,
            height_origin=height_origin,
            height_extent=height_extent,
            base=base,
            base_offset=base_offset,
        )

    # no_grad, not inference_mode: inference tensors recompile the fused norm.
    @torch.no_grad()
    def _run(
        self,
        low: torch.Tensor,
        *,
        probabilities,
        domain,
        seed,
        guidance,
        tile_size,
        overlap,
        margin,
        height_origin,
        height_extent,
        base,
        base_offset,
    ):
        """Shared refine path; tile_size, overlap and margin are HR voxels.

        Unset overlap is default_tiling's for tile_size.
        """
        if low.ndim not in (3, 4):
            raise ValueError(
                "LR input must be D,H,W labels or K,D,H,W phase fractions."
            )
        domain = self.generator.resolve_domain(domain)
        low_shape = tuple(low.shape[-3:])
        shape, margin, overlap, tiled = self._layout(
            low_shape, tile_size, overlap, margin
        )
        if base is None and base_offset is not None:
            raise ValueError("base_offset requires base.")
        if base is not None:
            base_offset = resolve_offset(base_offset, shape, volume_shape(base))
        guidance = self.generator.validate_guidance(guidance)
        # Height is validated on the LR grid the input volume was generated on.
        height_extent = self.generator.validate_height(
            low_shape, domain, height_origin, height_extent
        )
        check_sr_memory(
            estimate_sr_memory(
                low_shape,
                shape,
                self.num_phases,
                tile_size=tile_size,
                margin=margin,
                label_input=low.ndim == 3,
                input_element_size=low.element_size(),
                input_on_cuda=low.device.type == "cuda",
                probabilities=probabilities,
                base_shape=volume_shape(base),
            ),
            self.generator,
            input_device=low.device,
        )
        probs = fraction_channels(low, self.num_phases)
        height = (domain, height_origin, height_extent)
        with seeded_rng(seed, self.device):
            if not tiled and base is None:
                return self._refine_block(
                    probs, shape, margin, height, guidance, probabilities
                )
            return self._refine_tiled(
                probs,
                shape,
                tile_size if tiled else max(shape),
                overlap if tiled else 0,
                margin,
                height,
                guidance,
                probabilities,
                base,
                base_offset,
            )

    def _layout(self, low_shape, tile_size, overlap, margin):
        """Return the HR shape, resolved margin and overlap, and whether
        refinement is tiled."""
        shape = tuple(scaled_size(int(n), self.scale_factor) for n in low_shape)
        if margin is None:
            margin = self.default_margin
        if type(margin) is not int or margin < 0:
            raise ValueError("margin must be a non-negative integer of HR voxels.")
        if tile_size is not None:
            if type(tile_size) is not int or tile_size < 1:
                raise ValueError("tile_size must be a positive integer.")
            if overlap is None:
                overlap = self.default_tiling(tile_size)[1]
            check_overlap(tile_size, overlap)
        tiled = tile_size is not None and any(n > tile_size for n in shape)
        if tiled:
            if not self.scale_factor.is_integer():
                raise ValueError(
                    "tiled refinement requires an integer HR/LR scale; use one block for fractional scales."
                )
            scale = int(self.scale_factor)
            if any(
                value % scale
                for value in (*shape, tile_size, tile_size - 2 * overlap, margin)
            ):
                raise ValueError(
                    "HR shape, tile_size, stride and margin must be multiples of the LR/HR scale."
                )
        return shape, margin, overlap, tiled

    def _refine_block(self, probs, shape, margin, height, guidance, probabilities):
        domain, height_origin, height_extent = height
        coarse = resize_phases(probs, shape)
        coarse = F.pad(coarse, (margin,) * 6, mode="replicate").to(self.device)
        conditions = {
            "domain": self.generator.prepare_domain(domain),
            "coarse": coarse,
            "corruption_level": coarse.new_zeros(len(coarse)),
            "height": self.generator.height_condition(
                coarse.shape[-3:], domain, height_origin, -margin, height_extent
            ),
        }
        clean = self.generator.sample(torch.randn_like(coarse), conditions, guidance)
        region = (slice(None), *(slice(margin, margin + n) for n in shape))
        if not probabilities:
            return labels_from_channels(clean.squeeze(0)[region])
        # Convert and copy only the output, not the margin.
        probs = owned_clean_to_probs_(clean[(slice(None), *region)])
        return probs.squeeze(0).cpu().contiguous()

    def _refine_tiled(
        self,
        coarse,
        shape,
        tile_size,
        overlap,
        margin,
        height,
        guidance,
        probabilities,
        base,
        base_offset,
    ):
        domain, height_origin, height_extent = height
        # Include global context in the shared state. Each tile keeps its HR margin;
        # overlapping clean predictions are tapered before one posterior update.
        generation_shape = tuple(n + 2 * margin for n in shape)
        plan = replace(
            make_plan(generation_shape, tile_size + 2 * margin, overlap + margin),
            generation_shape=generation_shape,
            margin=margin,
        )
        tiles = make_tiles(plan)
        known = prepare_base(
            base, self.num_phases, shape, margin, base_offset, 0, base is not None
        )

        # A single block also supports fractional scale and arbitrary margin; its
        # condition is the same at every step, so resize it once.
        whole = (
            None
            if len(tiles) > 1
            else F.pad(
                resize_phases(coarse, shape), (margin,) * 6, mode="replicate"
            ).to(self.device)
        )

        def conditions(tile, index):
            origin = tuple(part.start - margin for part in tile.source)
            expanded = tuple(part.stop - part.start for part in tile.source)
            if whole is None:
                values = coarse_region(
                    coarse, origin, expanded, int(self.scale_factor)
                ).to(self.device)
            else:
                values = whole
            return {
                "coarse": values,
                "corruption_level": values.new_zeros(1),
                "height": self.generator.height_condition(
                    expanded, domain, height_origin, origin[0], height_extent
                ),
            }

        return TileSampler(self.generator).sample(
            tiles,
            plan,
            base=known,
            domain=self.generator.prepare_domain(domain),
            probabilities=probabilities,
            guidance=guidance,
            tile_conditions=conditions,
        )
