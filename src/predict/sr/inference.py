import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.build.model import build_diffusion, build_sr_model
from src.build.predict import resolve_weights
from src.config.train import get_sr_resolution, normalize_train_config
from src.device import check_device, default_device
from src.predict.convert import labels_from_channels, owned_clean_to_probs_
from src.predict.generator import Generator, GuidedDenoiser
from src.predict.sr.memory import check_sr_memory, estimate_sr_memory
from src.predict.sr.tiled import refine_tiled
from src.predict.tiling.layout import DEFAULT_OVERLAP, check_overlap
from src.predict.volume_condition import (
    fraction_channels,
    resolve_offset,
    volume_shape,
)
from src.prepare.height import height_field, resolve_extent
from src.prepare.resize import resize_phases, scaled_size
from src.random import seeded_rng


def load_sr_export(path: str | Path) -> dict:
    """Load an exported SR generator.pt and return it with its resolved config."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "diffusion-gan3d.sr":
        raise ValueError(
            "use exported SR weights (generator.pt), not a training or stage-1 checkpoint."
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
        self.model = build_sr_model(self.config).to(self.device)
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
            self.config["train"]["mixed_precision"] and self.device.type == "cuda",
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
        return self._resolve(
            low,
            output_kind="labels",
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
        return self._resolve(
            low,
            output_kind="probabilities",
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
    def _resolve(
        self,
        low: torch.Tensor,
        *,
        output_kind,
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
        domain_ids = self.generator.prepare_domain(domain)
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
        height_extent = self._validate_height(
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
                output_kind=output_kind,
                base_shape=volume_shape(base),
            ),
            self.generator,
            input_device=low.device,
        )
        probs = fraction_channels(low, self.num_phases)
        with seeded_rng(seed, self.device):
            if not tiled and base is None:
                return self._refine_block(
                    probs,
                    shape,
                    margin,
                    domain_ids,
                    height_origin,
                    height_extent,
                    guidance,
                    output_kind,
                )
            return refine_tiled(
                self,
                probs,
                shape,
                tile_size if tiled else max(shape),
                overlap if tiled else 0,
                margin,
                domain_ids,
                height_origin,
                guidance,
                output_kind,
                base,
                base_offset,
                height_extent,
            )

    def _validate_height(self, shape, domain, origin, extent=None):
        if not self.config["conditioning"]["height_enabled"]:
            if extent is not None:
                raise ValueError(
                    "height_extent requires a height-conditioned SR model."
                )
            return
        data = self.config["data"]
        extent = resolve_extent(data, domain, extent)
        maximum = extent - shape[0] * self.crop_size / self.lo_res_size
        if not math.isfinite(origin) or not 0 <= origin <= maximum:
            raise ValueError(
                "height_origin places the LR volume outside the source image height."
            )

        return extent

    def _refine_block(
        self,
        probs,
        shape,
        margin,
        domain,
        height_origin,
        height_extent,
        guidance,
        output_kind,
    ):
        coarse = resize_phases(probs, shape)
        coarse = F.pad(coarse, (margin,) * 6, mode="replicate")
        height = self._height(
            coarse.shape[-3:],
            (-margin,) * 3,
            int(domain.item()),
            height_origin,
            height_extent,
        )
        region = (slice(None), *(slice(margin, margin + n) for n in shape))
        clean = self._sample_clean(coarse, domain, height, guidance)
        if output_kind == "labels":
            return labels_from_channels(clean.squeeze(0)[region])
        probs = owned_clean_to_probs_(clean).squeeze(0).cpu()
        return probs[region].contiguous()

    def _height(self, shape, start, domain, origin, extent=None):
        if not self.config["conditioning"]["height_enabled"]:
            return None
        data = self.config["data"]
        spacing = self.crop_size / self.hi_res_size
        return height_field(
            shape,
            0,
            origin + start[0] * spacing,
            spacing,
            resolve_extent(data, domain, extent),
            self.device,
        )

    def _sample_clean(self, coarse, domain, height, guidance):
        coarse = coarse.to(self.device)
        conditions = {
            "domain": domain,
            "coarse": coarse,
            "corruption_level": coarse.new_zeros(len(coarse)),
            "height": height,
        }
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.generator.use_amp
        ):
            return self.generator.diffusion.sample(
                GuidedDenoiser(self.generator, guidance),
                torch.randn_like(coarse),
                self.generator.latent_channels,
                conditions,
            )
