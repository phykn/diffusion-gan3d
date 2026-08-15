import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .anchor import PlaneAnchor, build_anchors
from .diffusion import Diffusion
from .model.denoiser import Denoiser3D, validate_guidance


class _GuidedDenoiser:
    def __init__(self, generator: "Generator", guidance: float) -> None:
        self.generator = generator
        self.guidance = guidance

    def __call__(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
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


@dataclass(frozen=True)
class _SpatialAnchorAxis:
    axis: int
    weight: torch.Tensor


class _SpatialAnchorDenoiser:
    def __init__(
        self,
        generator: "Generator",
        guidance: float,
        anchor_image: torch.Tensor,
        anchor_mask: torch.Tensor,
        axes: tuple[_SpatialAnchorAxis, ...],
        residual_blur: float,
        residual_blur_early: float,
    ) -> None:
        self.generator = generator
        self.guidance = guidance
        self.anchor_image = anchor_image
        self.anchor_mask = anchor_mask
        self.axes = axes
        self.kernels = self.make_kernel_schedule(
            generator,
            axes,
            residual_blur,
            residual_blur_early,
        )

    def __call__(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plain = self.generator.predict_logits(
            current,
            time,
            latent,
            guidance=self.guidance,
            domain=domain,
            vf=vf,
        )
        baseline = plain.float()
        conditioned = self.generator.predict_logits(
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
        correction = torch.zeros_like(baseline)
        weight_sum = torch.zeros_like(self.axes[0].weight)
        max_weight = torch.zeros_like(weight_sum)
        for anchor_axis in self.axes:
            smoothed = self.smooth_scheduled(
                residual,
                time,
                anchor_axis.axis,
            )
            weight = anchor_axis.weight.to(device=plain.device, dtype=torch.float32)
            correction.add_(smoothed.mul(weight))
            weight_sum.add_(weight)
            max_weight = torch.maximum(max_weight, weight)
        correction.mul_(max_weight / weight_sum.clamp_min(torch.finfo(torch.float32).eps))
        if self.generator.diffusion.timesteps == 1:
            temporal_scale = torch.ones_like(time, dtype=torch.float32)
        else:
            temporal_scale = time.to(torch.float32).div(
                self.generator.diffusion.timesteps - 1
            )
            temporal_scale.mul_(0.75).add_(0.25).mul_(2.0)
        correction.mul_(temporal_scale.view(-1, 1, 1, 1, 1))
        logits = baseline.add_(correction)
        return Denoiser3D.decode(logits).to(current.dtype)

    def smooth_scheduled(
        self,
        residual: torch.Tensor,
        time: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        smoothed = torch.empty_like(residual)
        for step in time.unique():
            selected = time == step
            smoothed[selected] = self.smooth_axis(
                residual[selected],
                axis,
                self.kernels[int(step.item())][axis],
            ).to(residual.dtype)
        return smoothed

    @classmethod
    def make_kernel_schedule(
        cls,
        generator: "Generator",
        axes: tuple[_SpatialAnchorAxis, ...],
        residual_blur: float,
        residual_blur_early: float,
    ) -> dict[int, dict[int, torch.Tensor]]:
        last_step = generator.diffusion.timesteps - 1
        schedule = {}
        for step in range(generator.diffusion.timesteps):
            progress = 0.0 if last_step == 0 else step / last_step
            sigma = residual_blur + progress * (
                residual_blur_early - residual_blur
            )
            schedule[step] = cls.make_kernels(generator, axes, sigma)
        return schedule

    @staticmethod
    def make_kernels(
        generator: "Generator",
        axes: tuple[_SpatialAnchorAxis, ...],
        sigma: float,
    ) -> dict[int, torch.Tensor]:
        kernels = {}
        for axis in {item.axis for item in axes}:
            if sigma == 0.0:
                kernels[axis] = torch.ones(
                    (generator.num_phases, 1, 1, 1, 1),
                    device=generator.device,
                    dtype=torch.float32,
                )
                continue
            length = axes[0].weight.shape[axis + 2]
            radius = min(math.ceil(3.0 * sigma), max(1, (length - 1) // 2))
            positions = torch.arange(
                -radius,
                radius + 1,
                device=generator.device,
                dtype=torch.float32,
            )
            kernel = torch.exp(positions.square().mul(-0.5 / sigma**2))
            kernel.div_(kernel.sum())
            shape = [1, 1, 1, 1, 1]
            shape[axis + 2] = kernel.numel()
            kernels[axis] = kernel.view(shape).expand(
                generator.num_phases,
                1,
                *shape[2:],
            ).contiguous()
        return kernels

    @staticmethod
    def smooth_axis(
        residual: torch.Tensor,
        axis: int,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        radius = (kernel.shape[axis + 2] - 1) // 2
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - axis)] = radius
        padding[2 * (2 - axis) + 1] = radius
        padded = F.pad(residual, tuple(padding), mode="replicate")
        return F.conv3d(padded, kernel, groups=residual.shape[1])


class _CoupledAnchorSampler:
    def __init__(
        self,
        generator: "Generator",
        guidance: float,
        anchor_image: torch.Tensor,
        anchor_mask: torch.Tensor,
        axes: tuple[_SpatialAnchorAxis, ...],
        coupling_weight: torch.Tensor,
        residual_blur: float,
        residual_blur_early: float,
    ) -> None:
        self.generator = generator
        self.guidance = guidance
        self.anchor_denoiser = _SpatialAnchorDenoiser(
            generator,
            guidance,
            anchor_image,
            anchor_mask,
            axes,
            residual_blur,
            residual_blur_early,
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
        last_step = generator.diffusion.timesteps - 1
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
                generator.predict_logits(
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
            posterior_noise = (
                None if transition == 0 else torch.randn_like(base_state)
            )
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
            progress = 1.0 if last_step == 0 else 1.0 - transition / last_step
            release = progress**4
            coupling_weight = self.coupling_weight.lerp(
                torch.ones_like(self.coupling_weight),
                release,
            )
            anchor_state = torch.lerp(
                base_next.float(),
                anchor_next_raw.float(),
                coupling_weight,
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
    ) -> None:
        self.model = model
        self.num_domains = model.num_domains
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.use_amp = use_amp

    def predict(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        domain: torch.Tensor,
        guidance: float = 1.0,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        guidance = validate_guidance(guidance)
        conditions = {"domain": domain}
        if vf is not None:
            conditions["vf"] = vf
        if anchor_image is not None:
            conditions["anchor_image"] = anchor_image
        if anchor_mask is not None:
            conditions["anchor_mask"] = anchor_mask
        if guidance == 1.0:
            return self.model(current, time, latent, **conditions)
        return self.model.predict_guided(
            current,
            time,
            latent,
            guidance,
            **conditions,
        )

    def predict_logits(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        *,
        domain: torch.Tensor,
        guidance: float = 1.0,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        guidance = validate_guidance(guidance)
        conditions = {"domain": domain}
        if vf is not None:
            conditions["vf"] = vf
        if anchor_image is not None:
            conditions["anchor_image"] = anchor_image
        if anchor_mask is not None:
            conditions["anchor_mask"] = anchor_mask
        if guidance == 1.0:
            return self.model.predict_logits(current, time, latent, **conditions)
        return self.model.predict_guided_logits(
            current,
            time,
            latent,
            guidance,
            **conditions,
        )

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
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int = 8,
        anchor_sigma: float = 2.0,
        anchor_residual_blur: float = 1.5,
        anchor_residual_blur_early: float = 2.0,
    ) -> torch.Tensor:
        size = self.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
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
        if (
            not isinstance(anchor_sigma, (int, float))
            or isinstance(anchor_sigma, bool)
            or not math.isfinite(anchor_sigma)
            or anchor_sigma <= 0.0
        ):
            raise ValueError("anchor_sigma must be a positive finite number.")
        anchor_sigma = float(anchor_sigma)
        if (
            not isinstance(anchor_residual_blur, (int, float))
            or isinstance(anchor_residual_blur, bool)
            or not math.isfinite(anchor_residual_blur)
            or anchor_residual_blur < 0.0
        ):
            raise ValueError(
                "anchor_residual_blur must be a non-negative finite number."
            )
        anchor_residual_blur = float(anchor_residual_blur)
        if (
            not isinstance(anchor_residual_blur_early, (int, float))
            or isinstance(anchor_residual_blur_early, bool)
            or not math.isfinite(anchor_residual_blur_early)
            or anchor_residual_blur_early < 0.0
        ):
            raise ValueError(
                "anchor_residual_blur_early must be a non-negative finite number."
            )
        anchor_residual_blur_early = float(anchor_residual_blur_early)
        guidance = validate_guidance(guidance)
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
        anchor_axes: tuple[_SpatialAnchorAxis, ...] = ()
        coupling_weight = None
        if anchor_strength > 0.0:
            shifted_anchors = self.offset_anchors(anchors, size, margin)
            anchor = build_anchors(
                shifted_anchors,
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=generation_size,
                device=self.device,
                dtype=initial_noise.dtype,
            )
            if anchor is not None:
                grouped = []
                for axis in sorted({item.axis for item in shifted_anchors}):
                    axis_anchors = tuple(
                        item for item in shifted_anchors if item.axis == axis
                    )
                    grouped.append(
                        _SpatialAnchorAxis(
                            axis=axis,
                            weight=self.make_anchor_weight(
                                axis_anchors,
                                generation_size,
                                anchor_sigma,
                                anchor_strength,
                                device=self.device,
                            ),
                        )
                    )
                anchor_axes = tuple(grouped)
                coupling_weight = self.make_anchor_coupling_weight(
                    shifted_anchors,
                    generation_size,
                    inner_radius=anchor_sigma * 3.0,
                    outer_radius=anchor_sigma * 10.0,
                    device=self.device,
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
                assert coupling_weight is not None
                sampler = _CoupledAnchorSampler(
                    self,
                    guidance,
                    anchor.image,
                    anchor.mask,
                    anchor_axes,
                    coupling_weight,
                    anchor_residual_blur,
                    anchor_residual_blur_early,
                )
                clean = sampler.sample(initial_noise, conditions)
            else:
                sampling_model = (
                    self.model if guidance == 1.0 else _GuidedDenoiser(self, guidance)
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
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a Gaussian falloff around the spatial support of each anchor."""
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
            gaussian = torch.exp(distance_sq.mul(-0.5 / sigma**2)).mul(strength)
            weight = torch.maximum(weight, gaussian)
        return weight.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def make_anchor_coupling_weight(
        anchors: Sequence[PlaneAnchor],
        volume_size: int,
        inner_radius: float,
        outer_radius: float,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Keep the anchor trajectory locally, then taper to the baseline."""
        shape = (volume_size, volume_size, volume_size)
        coords = [
            torch.arange(volume_size, device=device, dtype=torch.float32).view(
                tuple(volume_size if dim == axis else 1 for dim in range(3))
            )
            for axis in range(3)
        ]
        weight = torch.zeros(shape, device=device, dtype=torch.float32)
        width = outer_radius - inner_radius
        for anchor in anchors:
            height, plane_width = anchor.image.shape[-2:]
            if anchor.position is None:
                row = (volume_size - height) // 2
                col = (volume_size - plane_width) // 2
            else:
                row, col = anchor.position
            remaining = [axis for axis in range(3) if axis != anchor.axis]
            distance_sq = (coords[anchor.axis] - anchor.index).square()
            for axis, start, length in zip(
                remaining,
                (row, col),
                (height, plane_width),
                strict=True,
            ):
                before = (start - coords[axis]).clamp_min(0.0)
                after = (coords[axis] - (start + length - 1)).clamp_min(0.0)
                distance_sq = distance_sq + (before + after).square()
            phase = (
                distance_sq.sqrt().sub(inner_radius).div(width).clamp_(0.0, 1.0)
            )
            taper = phase.mul(math.pi).cos_().add_(1.0).mul_(0.5)
            weight = torch.maximum(weight, taper)
        return weight.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def offset_anchors(
        anchors: Sequence[PlaneAnchor],
        size: int,
        margin: int,
    ) -> tuple[PlaneAnchor, ...]:
        anchors = tuple(anchors)
        if any(not isinstance(anchor, PlaneAnchor) for anchor in anchors):
            raise TypeError("anchors must contain only PlaneAnchor values.")
        for anchor in anchors:
            if (
                not isinstance(anchor.index, int)
                or isinstance(anchor.index, bool)
                or not 0 <= anchor.index < size
            ):
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
            if (
                isinstance(anchor.position, tuple)
                and len(anchor.position) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in anchor.position
                )
            ):
                row, col = anchor.position
                if row < 0 or col < 0 or row + height > size or col + width > size:
                    raise ValueError(
                        "anchor.position places the image outside the plane."
                    )
        if margin == 0:
            return anchors
        shifted = []
        for anchor in anchors:
            position = anchor.position
            if position is not None and (
                isinstance(position, tuple)
                and len(position) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in position
                )
            ):
                position = (position[0] + margin, position[1] + margin)
            shifted.append(
                PlaneAnchor(
                    image=anchor.image,
                    axis=anchor.axis,
                    index=anchor.index + margin,
                    position=position,
                )
            )
        return tuple(shifted)

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
        margin: int = 8,
        anchor_sigma: float = 2.0,
        anchor_residual_blur: float = 1.5,
        anchor_residual_blur_early: float = 2.0,
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            anchor_sigma=anchor_sigma,
            anchor_residual_blur=anchor_residual_blur,
            anchor_residual_blur_early=anchor_residual_blur_early,
            guidance=guidance,
            domain=domain,
            margin=margin,
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
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int = 8,
        anchor_sigma: float = 2.0,
        anchor_residual_blur: float = 1.5,
        anchor_residual_blur_early: float = 2.0,
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            anchor_sigma=anchor_sigma,
            anchor_residual_blur=anchor_residual_blur,
            anchor_residual_blur_early=anchor_residual_blur_early,
            guidance=guidance,
            domain=domain,
            margin=margin,
        )
        probs = clean.float()
        probs.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        return probs.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
