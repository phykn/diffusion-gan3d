import math
from collections.abc import Sequence

import torch

from .anchor import PlaneAnchor, build_anchors
from .diffusion import Diffusion
from .model.denoiser import Denoiser3D, validate_guidance_scale


class _GuidedDenoiser:
    def __init__(self, generator: "Generator", guidance_scale: float) -> None:
        self.generator = generator
        self.guidance_scale = guidance_scale

    def __call__(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.generator.predict(
            current,
            time,
            latent,
            guidance_scale=self.guidance_scale,
            vf=vf,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )


class Generator:
    def __init__(
        self,
        model: Denoiser3D,
        diffusion: Diffusion,
        device: torch.device,
        patch_size: int,
        num_phases: int,
        latent_channels: int,
        anchor_enabled: bool,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.anchor_enabled = anchor_enabled
        self.use_amp = use_amp

    def predict(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        guidance_scale: float = 1.0,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        guidance_scale = validate_guidance_scale(guidance_scale)
        conditions = {}
        if vf is not None:
            conditions["vf"] = vf
        if anchor_image is not None:
            conditions["anchor_image"] = anchor_image
        if anchor_mask is not None:
            conditions["anchor_mask"] = anchor_mask
        if guidance_scale == 1.0:
            return self.model(current, time, latent, **conditions)
        return self.model.predict_guided(
            current,
            time,
            latent,
            guidance_scale,
            **conditions,
        )

    def prepare_vf(
        self,
        vf: Sequence[float] | None,
    ) -> torch.Tensor | None:
        if vf is None:
            return None
        values = torch.as_tensor(
            vf,
            device=self.device,
            dtype=torch.float64,
        )
        if values.shape != (self.num_phases,):
            raise ValueError(f"vf must have shape [{self.num_phases}].")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("vf values must be finite.")
        if bool((values < 0).any()):
            raise ValueError("vf values must be non-negative.")
        scale = values.max()
        if scale == 0:
            raise ValueError("vf sum must not be zero.")
        values = values.div(scale)
        values.div_(values.sum())
        return values.to(torch.float32).unsqueeze(0)

    @torch.no_grad()
    def _sample_clean(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
        anchor_strength: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        size = self.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
        if (
            not isinstance(anchor_strength, (int, float))
            or isinstance(anchor_strength, bool)
            or not math.isfinite(anchor_strength)
            or not 0.0 <= anchor_strength <= 1.0
        ):
            raise ValueError("anchor_strength must be between zero and one.")
        anchor_strength = float(anchor_strength)
        guidance_scale = validate_guidance_scale(guidance_scale)
        vf = self.prepare_vf(vf)
        initial_noise = torch.randn(
            1,
            self.num_phases,
            size,
            size,
            size,
            device=self.device,
            dtype=torch.float32,
        )
        anchor = None
        if anchor_strength > 0.0:
            anchor = build_anchors(
                anchors,
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=size,
                device=self.device,
                dtype=initial_noise.dtype,
            )
        if anchor is not None and not self.anchor_enabled:
            raise ValueError("selected weights were trained with anchors disabled.")

        conditions = {}
        known_clean: torch.Tensor | None = None
        known_mask: torch.Tensor | None = None
        if anchor is not None:
            anchor_mask = anchor.mask
            if anchor_strength != 1.0:
                anchor_mask = anchor_mask.to(initial_noise.dtype).mul(anchor_strength)
            conditions.update(
                {
                    "anchor_image": anchor.image,
                    "anchor_mask": anchor_mask,
                }
            )
            known_clean = anchor.image
            known_mask = anchor_mask
        if vf is not None:
            conditions["vf"] = vf
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            sampling_model = (
                self.model
                if guidance_scale == 1.0
                else _GuidedDenoiser(self, guidance_scale)
            )
            clean = self.diffusion.sample(
                sampling_model,
                initial_noise,
                self.latent_channels,
                conditions=conditions or None,
                known_clean=known_clean,
                known_mask=known_mask,
            )
        return clean

    def generate_probs(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
        anchor_strength: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            guidance_scale=guidance_scale,
        )
        probs = (clean.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)
        probs.div_(
            probs.sum(dim=1, keepdim=True).clamp_min_(torch.finfo(probs.dtype).eps)
        )
        return probs.squeeze(0).cpu()

    def generate(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
        anchor_strength: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            guidance_scale=guidance_scale,
        )
        probs = clean.float()
        probs.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        return probs.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
