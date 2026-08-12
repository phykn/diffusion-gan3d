import math
from collections.abc import Sequence

import torch

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
        if anchor_strength > 0.0:
            anchor = build_anchors(
                self.offset_anchors(anchors, size, margin),
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=generation_size,
                device=self.device,
                dtype=initial_noise.dtype,
            )
        conditions = {"domain": self.prepare_domain(domain)}
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
        if vf is not None:
            conditions["vf"] = vf
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
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
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
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
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            guidance=guidance,
            domain=domain,
            margin=margin,
        )
        probs = clean.float()
        probs.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        return probs.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
