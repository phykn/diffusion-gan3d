import math
from collections.abc import Sequence

import torch

from ..anchor import PlaneAnchor, encode_anchors
from .denoiser import Denoiser3D
from .diffusion import Diffusion


class GuidedDenoiser:
    def __init__(self, generator: "Generator", guidance: float) -> None:
        self.generator = generator
        self.guidance = guidance

    def __call__(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.generator.predict(
            current,
            time,
            latent,
            guidance=self.guidance,
            domain=domain,
            vf=vf,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )


class SpatialAnchorDenoiser:
    def __init__(
        self,
        generator: "Generator",
        guidance: float,
        anchor_image: torch.Tensor,
        anchor_mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        self.generator = generator
        self.guidance = guidance
        self.anchor_image = anchor_image
        self.anchor_mask = anchor_mask
        self.weight = weight

    def __call__(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plain = self.generator.compute_logits(
            current,
            time,
            latent,
            guidance=self.guidance,
            domain=domain,
            vf=vf,
        )
        baseline = plain.float()
        conditioned = self.generator.compute_logits(
            current,
            time,
            latent,
            guidance=self.guidance,
            domain=domain,
            vf=vf,
            anchor_image=self.anchor_image,
            anchor_mask=self.anchor_mask,
        )
        residual = conditioned.float().sub(baseline)
        weight = self.weight.to(device=plain.device, dtype=torch.float32)
        correction = residual * weight
        plane_scale, context_scale = self.compute_temporal_scales(
            time,
            self.generator.diffusion.alpha_bars,
        )
        plane_scale = plane_scale.view(-1, 1, 1, 1, 1)
        context_scale = context_scale.view(-1, 1, 1, 1, 1)
        temporal_scale = torch.lerp(
            context_scale,
            plane_scale,
            self.anchor_mask.to(torch.float32),
        )
        correction = correction * temporal_scale
        logits = baseline + correction
        return Denoiser3D.decode(logits).to(current.dtype)

    @staticmethod
    def compute_temporal_scales(
        time: torch.Tensor,
        alpha_bars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = alpha_bars.numel() - 1
        if timesteps == 1:
            progress = torch.ones_like(time, dtype=torch.float32)
        else:
            progress = 1.0 - time.to(torch.float32) / (timesteps - 1)
        plane = 0.5 + 0.5 * progress.square()
        alpha_bars = alpha_bars.to(device=time.device, dtype=torch.float32)
        initial = torch.sqrt(torch.full_like(progress, 2.0))
        final = (1.0 - alpha_bars[1]).clamp_min(0.0).sqrt()
        context = torch.lerp(initial, final, progress)
        return plane, context


class CoupledAnchorSampler:
    def __init__(
        self,
        generator: "Generator",
        guidance: float,
        anchor_image: torch.Tensor,
        anchor_mask: torch.Tensor,
        weight: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> None:
        self.generator = generator
        self.guidance = guidance
        self.anchor_denoiser = SpatialAnchorDenoiser(
            generator,
            guidance,
            anchor_image,
            anchor_mask,
            weight,
        )
        self.coupling_weight = coupling_weight

    def sample(
        self,
        initial_noise: torch.Tensor,
        conditions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generator = self.generator
        base_state = initial_noise.clone()
        anchor_state = initial_noise.clone()
        for transition in reversed(range(generator.diffusion.timesteps)):
            time = torch.full(
                (initial_noise.shape[0],),
                transition,
                device=generator.device,
                dtype=torch.long,
            )
            latent = torch.randn(
                initial_noise.shape[0],
                generator.latent_channels,
                device=generator.device,
                dtype=initial_noise.dtype,
            )
            base_pred = Denoiser3D.decode(
                generator.compute_logits(
                    base_state,
                    time,
                    latent,
                    guidance=self.guidance,
                    **conditions,
                )
            ).to(base_state.dtype)
            anchor_pred = self.anchor_denoiser(
                anchor_state,
                time,
                latent,
                **conditions,
            )
            posterior_noise = None if transition == 0 else torch.randn_like(base_state)
            base_next = generator.diffusion.sample_posterior(
                base_state,
                base_pred,
                transition,
                noise=posterior_noise,
            )
            anchor_next_raw = generator.diffusion.sample_posterior(
                anchor_state,
                anchor_pred,
                transition,
                noise=posterior_noise,
            )
            anchor_state = torch.lerp(
                base_next.float(),
                anchor_next_raw.float(),
                self.coupling_weight,
            ).to(base_next.dtype)
            base_state = base_next
        return anchor_state


class Generator:
    def __init__(
        self,
        model: Denoiser3D,
        diffusion: Diffusion,
        device: torch.device,
        patch_size: int,
        num_phases: int,
        latent_channels: int,
        use_amp: bool,
        anchor_spread: float | None = None,
    ) -> None:
        self.model = model
        self.num_domains = model.num_domains
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.use_amp = use_amp
        if anchor_spread is not None and (
            not isinstance(anchor_spread, (int, float))
            or isinstance(anchor_spread, bool)
            or not math.isfinite(anchor_spread)
            or anchor_spread <= 0.0
        ):
            raise ValueError("anchor_spread must be positive and finite.")
        self.default_anchor_spread = (
            None if anchor_spread is None else float(anchor_spread)
        )
        factor = getattr(model, "downsample_factor", None)
        if factor is None:
            if isinstance(model, Denoiser3D):
                raise AttributeError("Denoiser3D must expose downsample_factor.")
            factor = 1
        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
            raise ValueError("model.downsample_factor must be a positive integer.")
        self.default_margin = factor
        self.default_anchor_sigma = (
            self.patch_size * self.default_anchor_spread
            if self.default_anchor_spread is not None
            else math.sqrt(3.0) * factor
        )

    def predict(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        guidance: float = 1.0,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditions = self.prepare_conditions(domain, vf, anchor_image, anchor_mask)
        if guidance == 1.0:
            return self.model(current, time, latent, **conditions)
        logits = self.compute_logits(
            current,
            time,
            latent,
            guidance=guidance,
            **conditions,
        )
        return Denoiser3D.decode(logits).to(current.dtype)

    def compute_logits(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        guidance: float = 1.0,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditions = self.prepare_conditions(domain, vf, anchor_image, anchor_mask)
        if guidance == 1.0:
            return self.model.compute_logits(current, time, latent, **conditions)
        return self.model.apply_guidance_logits(
            current,
            time,
            latent,
            guidance,
            **conditions,
        )

    @staticmethod
    def prepare_conditions(
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conditions = {"domain": domain}
        if vf is not None:
            conditions["vf"] = vf
        if anchor_image is not None:
            conditions["anchor_image"] = anchor_image
        if anchor_mask is not None:
            conditions["anchor_mask"] = anchor_mask
        return conditions

    def prepare_domain(self, domain: int | None) -> torch.Tensor:
        if domain is None:
            if self.num_domains != 1:
                raise ValueError("domain is required for a multi-domain model.")
            domain = 0
        if (
            not isinstance(domain, int)
            or isinstance(domain, bool)
            or not 0 <= domain < self.num_domains
        ):
            raise ValueError(
                f"domain must be an integer from 0 to {self.num_domains - 1}."
            )
        return torch.tensor(
            (domain,),
            device=self.device,
            dtype=torch.long,
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
        total = values.sum()
        if total == 0:
            raise ValueError("vf sum must not be zero.")
        return (values / total).to(torch.float32).unsqueeze(0)

    @torch.no_grad()
    def _sample_clean(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
        anchor_strength: float = 1.0,
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int | None = None,
        anchor_sigma: float | None = None,
    ) -> torch.Tensor:
        size = self.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
        margin = self.default_margin if margin is None else margin
        if not isinstance(margin, int) or isinstance(margin, bool) or margin < 0:
            raise ValueError("margin must be a non-negative integer.")
        if (
            not isinstance(anchor_strength, (int, float))
            or isinstance(anchor_strength, bool)
            or not math.isfinite(anchor_strength)
            or not 0.0 <= anchor_strength <= 1.0
        ):
            raise ValueError("anchor_strength must be between zero and one.")
        anchor_strength = float(anchor_strength)
        if anchor_sigma is None:
            anchor_sigma = (
                size * self.default_anchor_spread
                if self.default_anchor_spread is not None
                else self.default_anchor_sigma
            )
        if (
            not isinstance(anchor_sigma, (int, float))
            or isinstance(anchor_sigma, bool)
            or not math.isfinite(anchor_sigma)
            or anchor_sigma <= 0.0
        ):
            raise ValueError("anchor_sigma must be a positive finite number.")
        anchor_sigma = float(anchor_sigma)
        vf = self.prepare_vf(vf)
        generation_size = size + 2 * margin
        initial_noise = torch.randn(
            1,
            self.num_phases,
            generation_size,
            generation_size,
            generation_size,
            device=self.device,
            dtype=torch.float32,
        )
        anchor = None
        anchor_weight = None
        coupling_weight = None
        if anchor_strength > 0.0:
            shifted_anchors = self.offset_anchors(anchors, size, margin)
            anchor = encode_anchors(
                shifted_anchors,
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=generation_size,
                device=self.device,
                dtype=initial_noise.dtype,
            )
            if anchor is not None:
                anchor_weight = self.make_anchor_weight(
                    shifted_anchors,
                    generation_size,
                    anchor_sigma,
                    anchor_strength,
                    device=self.device,
                )
                coupling_weight = self.make_anchor_coupling_weight(
                    anchor_weight,
                    anchor_strength,
                )
        conditions = {"domain": self.prepare_domain(domain)}
        if vf is not None:
            conditions["vf"] = vf
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            if anchor is not None:
                assert anchor_weight is not None
                assert coupling_weight is not None
                sampler = CoupledAnchorSampler(
                    self,
                    guidance,
                    anchor.image,
                    anchor.mask,
                    anchor_weight,
                    coupling_weight,
                )
                clean = sampler.sample(initial_noise, conditions)
            else:
                sampling_model = (
                    self.model if guidance == 1.0 else GuidedDenoiser(self, guidance)
                )
                clean = self.diffusion.sample(
                    sampling_model,
                    initial_noise,
                    self.latent_channels,
                    conditions=conditions or None,
                )
        return self.crop_clean(clean, size, margin)

    @staticmethod
    def make_anchor_weight(
        anchors: Sequence[PlaneAnchor],
        volume_size: int,
        sigma: float,
        strength: float,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (volume_size, volume_size, volume_size)
        coords = [
            torch.arange(volume_size, device=device, dtype=torch.float32).view(
                tuple(volume_size if dim == axis else 1 for dim in range(3))
            )
            for axis in range(3)
        ]
        weight = torch.zeros(shape, device=device, dtype=torch.float32)
        for anchor in anchors:
            height, width = anchor.image.shape[-2:]
            if anchor.position is None:
                row = (volume_size - height) // 2
                col = (volume_size - width) // 2
            else:
                row, col = anchor.position
            remaining = [axis for axis in range(3) if axis != anchor.axis]
            distance_sq = (coords[anchor.axis] - anchor.index).square()
            for axis, start, length in zip(
                remaining,
                (row, col),
                (height, width),
                strict=True,
            ):
                before = (start - coords[axis]).clamp_min(0.0)
                after = (coords[axis] - (start + length - 1)).clamp_min(0.0)
                distance_sq = distance_sq + (before + after).square()
            gaussian = torch.exp(distance_sq * (-0.5 / sigma**2)) * strength
            weight = torch.maximum(weight, gaussian)
        return weight.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def make_anchor_coupling_weight(
        anchor_weight: torch.Tensor,
        anchor_strength: float,
    ) -> torch.Tensor:
        return (anchor_weight / anchor_strength).clamp(0.0, 1.0)

    @staticmethod
    def offset_anchors(
        anchors: Sequence[PlaneAnchor],
        size: int,
        margin: int,
    ) -> tuple[PlaneAnchor, ...]:
        anchors = tuple(anchors)
        for anchor in anchors:
            if not 0 <= anchor.index < size:
                raise ValueError("anchor.index is outside the generated volume.")
            if not isinstance(anchor.image, torch.Tensor) or anchor.image.ndim not in {
                2,
                3,
            }:
                continue
            height, width = anchor.image.shape[-2:]
            if height > size or width > size:
                raise ValueError("anchor.image must fit inside the generated plane.")
            if anchor.position is None:
                continue
            row, col = anchor.position
            if row < 0 or col < 0 or row + height > size or col + width > size:
                raise ValueError("anchor.position places the image outside the plane.")
        if margin == 0:
            return anchors
        return tuple(
            PlaneAnchor(
                image=anchor.image,
                axis=anchor.axis,
                index=anchor.index + margin,
                position=(
                    None
                    if anchor.position is None
                    else (
                        anchor.position[0] + margin,
                        anchor.position[1] + margin,
                    )
                ),
            )
            for anchor in anchors
        )

    @staticmethod
    def crop_clean(
        clean: torch.Tensor,
        size: int,
        margin: int,
    ) -> torch.Tensor:
        if margin == 0:
            return clean
        region = slice(margin, margin + size)
        return clean[:, :, region, region, region].clone()

    def generate_probs(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
        anchor_strength: float = 1.0,
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int | None = None,
        anchor_sigma: float | None = None,
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            anchor_sigma=anchor_sigma,
            guidance=guidance,
            domain=domain,
            margin=margin,
        )
        probs = ((clean.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).eps,
        )
        return probs.squeeze(0).cpu()

    def generate(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
        anchor_strength: float = 1.0,
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int | None = None,
        anchor_sigma: float | None = None,
    ) -> torch.Tensor:
        return self.generate_probs(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            guidance=guidance,
            domain=domain,
            margin=margin,
            anchor_sigma=anchor_sigma,
        ).argmax(dim=0).to(torch.uint8)
