import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.build.model import build_diffusion, build_sr_model
from src.config.train import get_sr_sizes, normalize_train_config
from src.predict.base import resolve_offset, volume_shape
from src.predict.convert import labels_from_channels, owned_clean_to_probs_
from src.predict.generator import Generator, GuidedDenoiser
from src.predict.random import seeded_rng
from src.predict.sr.memory import check_sr_memory, estimate_sr_memory
from src.predict.sr.tiled import refine_tiled
from src.prepare.height import height_field, resolve_extent
from src.prepare.resize import phase_channels, resize_phases, scaled_size


class SuperResolutionAPI:
    def __init__(self, weights: str | Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.weights = Path(weights).resolve()
        if self.weights.is_dir():
            self.weights = self.weights / "generator.pt"
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.sr":
            raise ValueError(
                "use exported SR weights (generator.pt), not a training or stage-1 checkpoint."
            )
        self.config = normalize_train_config(payload["config"], "sr")
        self.model = build_sr_model(self.config).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval().requires_grad_(False)
        self.crop_size, self.lo_res_size, self.hi_res_size = get_sr_sizes(self.config)
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

    def predict_probs(
        self,
        low: torch.Tensor,
        domain: int | None = None,
        seed: int = 0,
        tile_size: int | None = None,
        overlap: int = 8,
        height_origin: float = 0.0,
        margin: int | None = None,
        guidance: float = 1.0,
        *,
        base: torch.Tensor | None = None,
        base_offset=None,
        height_extent=None,
    ) -> torch.Tensor:
        return self._resolve(
            low,
            domain,
            seed,
            tile_size,
            overlap,
            height_origin,
            margin,
            guidance,
            output_kind="probabilities",
            base=base,
            base_offset=base_offset,
            height_extent=height_extent,
        )

    @torch.inference_mode()
    def _resolve(
        self,
        low,
        domain,
        seed,
        tile_size,
        overlap,
        height_origin,
        margin,
        guidance,
        *,
        output_kind,
        base=None,
        base_offset=None,
        height_extent=None,
    ):
        if low.ndim not in (3, 4):
            raise ValueError(
                "LR input must be D,H,W labels or K,D,H,W phase fractions."
            )
        domain_ids = self.generator.prepare_domain(domain)
        domain = int(domain_ids.item())
        low_shape = low.shape[-3:]
        shape = tuple(scaled_size(int(n), self.scale_factor) for n in low_shape)
        if base is None and base_offset is not None:
            raise ValueError("base_offset requires base.")
        if base is not None:
            base_offset, _ = resolve_offset(base_offset, shape, volume_shape(base))
        guidance = self.generator.validate_guidance(guidance)
        if margin is None:
            margin = self.generator.default_margin
            if self.scale_factor.is_integer():
                margin = math.lcm(margin, int(self.scale_factor))
        if type(margin) is not int or margin < 0:
            raise ValueError("margin must be a non-negative integer of HR voxels.")
        if tile_size is not None and (
            type(tile_size) is not int
            or tile_size < 1
            or type(overlap) is not int
            or overlap < 0
            or 2 * overlap >= tile_size
        ):
            raise ValueError(
                "tile_size must be positive and 0 <= 2 * overlap < tile_size."
            )
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
        probs = self._prepare_probs(
            low, shape, tile_size, margin, output_kind, volume_shape(base)
        )
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

    def _prepare_probs(self, low, shape, tile_size, margin, output_kind, base_shape):
        check_sr_memory(
            estimate_sr_memory(
                low.shape[-3:],
                shape,
                self.num_phases,
                tile_size=tile_size,
                margin=margin,
                label_input=low.ndim == 3,
                input_element_size=low.element_size(),
                input_on_cuda=low.device.type == "cuda",
                output_kind=output_kind,
                base_shape=base_shape,
            ),
            self.generator,
            input_device=low.device,
        )
        if low.ndim == 3:
            return phase_channels(low.cpu().unsqueeze(0), self.num_phases)
        if (
            not low.dtype.is_floating_point
            or low.shape[0] != self.num_phases
            or not torch.isfinite(low).all()
            or (low < 0).any()
            or (low > 1).any()
            or not torch.allclose(low.sum(0), torch.ones_like(low[0]), atol=1e-5)
        ):
            raise ValueError(
                "LR phase fractions must be finite, non-negative and sum to one."
            )
        return low.to(device="cpu", dtype=torch.float32).unsqueeze(0)

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
        if output_kind == "labels":
            clean = self._sample_clean(coarse, domain, height, guidance)
            return labels_from_channels(clean.squeeze(0)[region])
        predicted = self._predict(coarse, domain, height, guidance)
        return predicted[region].contiguous()

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
        }
        if height is not None:
            conditions["height"] = height
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.generator.use_amp
        ):
            clean = self.generator.diffusion.sample(
                GuidedDenoiser(self.generator, guidance),
                torch.randn_like(coarse),
                self.generator.latent_channels,
                conditions,
            )
        return clean

    def _predict(self, coarse, domain, height, guidance):
        clean = self._sample_clean(coarse, domain, height, guidance)
        probs = owned_clean_to_probs_(clean)
        return probs.squeeze(0).cpu()

    def super_resolve(
        self,
        low: torch.Tensor,
        domain: int | None = None,
        seed: int = 0,
        tile_size: int | None = None,
        overlap: int = 8,
        height_origin: float = 0.0,
        margin: int | None = None,
        guidance: float = 1.0,
        *,
        base: torch.Tensor | None = None,
        base_offset=None,
        height_extent=None,
    ) -> torch.Tensor:
        return self._resolve(
            low,
            domain,
            seed,
            tile_size,
            overlap,
            height_origin,
            margin,
            guidance,
            output_kind="labels",
            base=base,
            base_offset=base_offset,
            height_extent=height_extent,
        )
