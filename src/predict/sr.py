from fractions import Fraction
from itertools import product
from pathlib import Path

import torch

from src.build.model import build_sr_model
from src.config import get_domains, get_sr_sizes, normalize_train_config
from src.prepare.resize import phase_channels, scaled_size


class SuperResolutionAPI:
    def __init__(self, weights: str | Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.weights = Path(weights).resolve()
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.sr.v1":
            raise ValueError(
                "use exported SR weights/model.pt, not a stage-1 or training checkpoint."
            )
        self.config = normalize_train_config(payload["config"], "sr")
        self.model = build_sr_model(self.config).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval().requires_grad_(False)
        self.crop_size, self.lo_res_size, self.hi_res_size = get_sr_sizes(self.config)
        self.scale_factor = self.model.scale_factor

    @torch.inference_mode()
    def predict_probs(
        self,
        low: torch.Tensor,
        domain: int | None = None,
        seed: int = 0,
        tile_size: int | None = None,
        overlap: int = 8,
    ) -> torch.Tensor:
        if low.ndim != 3:
            raise ValueError("LR input must be a D,H,W phase-label volume.")
        domains = get_domains(self.config["data"])
        if domain is None:
            if len(domains) != 1:
                raise ValueError("domain is required for a multi-domain SR model.")
            domain = 0
        if (
            isinstance(domain, bool)
            or not isinstance(domain, int)
            or domain not in domains
        ):
            raise ValueError("invalid SR domain.")
        shape = tuple(scaled_size(int(n), self.scale_factor) for n in low.shape)
        probs = phase_channels(low.cpu().unsqueeze(0), self.model.num_phases)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(1, self.model.noise_channels, *low.shape, generator=rng)
        domain_ids = torch.tensor([domain], device=self.device)
        if tile_size is None:
            return self._predict(probs, noise, domain_ids).squeeze(0)
        if (
            isinstance(tile_size, bool)
            or not isinstance(tile_size, int)
            or tile_size < 1
            or isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or 2 * overlap >= tile_size
        ):
            raise ValueError(
                "tile_size must be positive and 0 <= 2 * overlap < tile_size."
            )
        lattice = Fraction(str(self.scale_factor)).denominator
        if tile_size % lattice or overlap % lattice:
            raise ValueError(
                f"tile_size and overlap must be multiples of {lattice} for this scale_factor."
            )
        lengths = tuple(min(tile_size, int(n)) for n in low.shape)
        starts = []
        for length, total in zip(lengths, low.shape, strict=True):
            stride = max(lattice, length - 2 * overlap)
            values = list(range(0, int(total) - length + 1, stride))
            if values[-1] != total - length:
                values.append(int(total) - length)
            starts.append(values)
        result = torch.zeros(self.model.num_phases, *shape)
        weights = torch.zeros(shape)
        high_lengths = tuple(scaled_size(n, self.scale_factor) for n in lengths)
        windows = [
            torch.hann_window(n, periodic=False).clamp_min(0.01)
            if n > 1
            else torch.ones(1)
            for n in high_lengths
        ]
        window = (
            windows[0][:, None, None]
            * windows[1][None, :, None]
            * windows[2][None, None, :]
        )
        for start in product(*starts):
            source = tuple(slice(s, s + n) for s, n in zip(start, lengths, strict=True))
            high_start = tuple(round(s * self.scale_factor) for s in start)
            target = tuple(
                slice(s, s + n) for s, n in zip(high_start, high_lengths, strict=True)
            )
            region = (slice(None), slice(None), *source)
            predicted = self._predict(probs[region], noise[region], domain_ids).squeeze(
                0
            )
            result[(slice(None), *target)] += predicted * window
            weights[target] += window
        return result / weights.unsqueeze(0)

    def _predict(self, low, noise, domain):
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = self.model(low.to(self.device), noise.to(self.device), domain)
        return logits.float().softmax(1).cpu()

    def super_resolve(
        self,
        low: torch.Tensor,
        domain: int | None = None,
        seed: int = 0,
        tile_size: int | None = None,
        overlap: int = 8,
    ) -> torch.Tensor:
        return (
            self.predict_probs(low, domain, seed, tile_size, overlap)
            .argmax(0)
            .to(torch.uint8)
        )
