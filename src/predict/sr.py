import math
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F

from src.build.model import build_diffusion, build_sr_model
from src.config import get_sr_sizes, normalize_train_config
from src.predict.generator import Generator, GuidedDenoiser
from src.predict.inference import _seeded_rng
from src.predict.tile import axis_starts
from src.prepare.height import height_field
from src.prepare.resize import coarse_region, phase_channels, resize_phases, scaled_size


class SuperResolutionAPI:
    def __init__(self, weights: str | Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.weights = Path(weights).resolve()
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.sr":
            raise ValueError(
                "use exported SR weights/model.pt, not a training or stage-1 checkpoint."
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

    @torch.inference_mode()
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
    ) -> torch.Tensor:
        """Refine a global LR volume. Tile size, overlap and margin use HR voxels."""
        if low.ndim not in (3, 4):
            raise ValueError(
                "LR input must be D,H,W labels or K,D,H,W phase fractions."
            )
        domain_ids = self.generator.prepare_domain(domain)
        domain = int(domain_ids.item())
        low_shape = low.shape[-3:]
        shape = tuple(scaled_size(int(n), self.scale_factor) for n in low_shape)
        if low.ndim == 3:
            probs = phase_channels(low.cpu().unsqueeze(0), self.num_phases)
        else:
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
            probs = low.float().cpu().unsqueeze(0)
        if not math.isfinite(guidance):
            raise ValueError("guidance must be finite.")
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
        self._validate_height(low_shape, domain, height_origin)
        with _seeded_rng(seed, self.device):
            if not tiled:
                coarse = resize_phases(probs, shape)
                coarse = F.pad(coarse, (margin,) * 6, mode="replicate")
                height = self._height(
                    coarse.shape[-3:], (-margin,) * 3, domain, height_origin
                )
                predicted = self._predict(coarse, domain_ids, height, guidance)
                region = (slice(None), *(slice(margin, margin + n) for n in shape))
                return predicted[region].contiguous()
            lengths = tuple(min(tile_size, n) for n in shape)
            starts = [
                axis_starts(n, length, tile_size - 2 * overlap)
                for n, length in zip(shape, lengths)
            ]
            result = torch.zeros(self.num_phases, *shape)
            weights = torch.zeros(shape)
            windows = [
                torch.hann_window(n, periodic=False).clamp_min(0.01)
                if n > 1
                else torch.ones(1)
                for n in lengths
            ]
            window = (
                windows[0][:, None, None]
                * windows[1][None, :, None]
                * windows[2][None, None, :]
            )
            expanded = tuple(n + 2 * margin for n in lengths)
            crop = (slice(None), *(slice(margin, margin + n) for n in lengths))
            for start in product(*starts):
                origin = tuple(s - margin for s in start)
                # Interpolation halo is separate from the denoiser's context margin.
                coarse = coarse_region(probs, origin, expanded, scale)
                height = self._height(expanded, origin, domain, height_origin)
                predicted = self._predict(coarse, domain_ids, height, guidance)[crop]
                target = tuple(slice(s, s + n) for s, n in zip(start, lengths))
                result[(slice(None), *target)] += predicted * window
                weights[target] += window
            return result / weights.unsqueeze(0)

    def _validate_height(self, shape, domain, origin):
        if not self.config["conditioning"]["height_enabled"]:
            return
        data = self.config["data"]
        axis = {"z": 0, "y": 1, "x": 2}[data["thickness_axis"]]
        maximum = (
            data["height_extents"][domain]
            - shape[axis] * self.crop_size / self.lo_res_size
        )
        if not math.isfinite(origin) or not 0 <= origin <= maximum:
            raise ValueError(
                "height_origin places the LR volume outside the measured thickness."
            )

    def _height(self, shape, start, domain, origin):
        if not self.config["conditioning"]["height_enabled"]:
            return None
        data = self.config["data"]
        axis = {"z": 0, "y": 1, "x": 2}[data["thickness_axis"]]
        spacing = self.crop_size / self.hi_res_size
        return height_field(
            shape,
            axis,
            origin + start[axis] * spacing,
            spacing,
            data["height_extents"][domain],
            self.device,
        )

    def _predict(self, coarse, domain, height, guidance):
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
        probs = ((clean.float() + 1) * 0.5).clamp(0, 1)
        probs = probs / probs.sum(1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).eps
        )
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
    ) -> torch.Tensor:
        return (
            self.predict_probs(
                low, domain, seed, tile_size, overlap, height_origin, margin, guidance
            )
            .argmax(0)
            .to(torch.uint8)
        )
