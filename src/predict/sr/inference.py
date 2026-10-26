import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from src.build.model import build_denoiser, build_diffusion
from src.build.predict import resolve_weights
from src.config.train import get_sr_resolution, normalize_train_config
from src.device import check_device, default_device, inference_amp
from src.predict.convert import labels_from_channels, owned_clean_to_probs_
from src.predict.generator import Generator
from src.predict.sr.memory import check_sr_memory, estimate_sr_memory
from src.predict.tiling.layout import (
    DEFAULT_OVERLAP,
    check_overlap,
    make_plan,
    make_tiles,
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
        self.device = check_device(default_device() if device is None else device)
        self.weights = resolve_weights(weights)
        payload = load_sr_export(self.weights)
        self.config = payload["config"]
        self.model = build_denoiser(self.config).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval().requires_grad_(False)
        resolution = get_sr_resolution(self.config)
        self.crop_size = resolution.crop_size
        self.lo_res_size = resolution.lo_res_size
        self.hi_res_size = resolution.hi_res_size
        self.scale_factor = self.hi_res_size / self.lo_res_size
        self.num_phases = self.config["data"]["num_phases"]
        self.generator = Generator(
            self.model,
            build_diffusion(self.config).to(self.device),
            self.device,
            self.hi_res_size,
            self.num_phases,
            self.config["model"]["generator"]["latent_channels"],
            inference_amp(self.config["train"]["mixed_precision"], self.device),
            height_data=self.config["data"]
            if self.config["conditioning"]["height_enabled"]
            else None,
        )

    def refine(
        self,
        low: torch.Tensor,
        *,
        domain: int | None = None,
        seed: int | None = None,
        guidance: float = 1.0,
        tile_size: int | None = None,
        overlap: int = DEFAULT_OVERLAP,
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
        overlap: int = DEFAULT_OVERLAP,
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

    @torch.inference_mode()
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
        """Shared refine path; tile_size, overlap and margin are HR voxels."""
        if low.ndim not in (3, 4):
            raise ValueError(
                "LR input must be D,H,W labels or K,D,H,W phase fractions."
            )
        domain = self.generator.resolve_domain(domain)
        low_shape = low.shape[-3:]
        shape = tuple(scaled_size(int(n), self.scale_factor) for n in low_shape)
        if base is None and base_offset is not None:
            raise ValueError("base_offset requires base.")
        if base is not None:
            base_offset = resolve_offset(base_offset, shape, volume_shape(base))
        guidance = self.generator.validate_guidance(guidance)
        if margin is None:
            margin = self.generator.default_margin
            if self.scale_factor.is_integer():
                margin = math.lcm(margin, int(self.scale_factor))
        if type(margin) is not int or margin < 0:
            raise ValueError("margin must be a non-negative integer of HR voxels.")
        if tile_size is not None:
            if type(tile_size) is not int or tile_size < 1:
                raise ValueError("tile_size must be a positive integer.")
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
        probs = owned_clean_to_probs_(clean).squeeze(0).cpu()
        return probs[region].contiguous()

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
            vf=None,
            domain=self.generator.prepare_domain(domain),
            probabilities=probabilities,
            guidance=guidance,
            tile_conditions=conditions,
        )
