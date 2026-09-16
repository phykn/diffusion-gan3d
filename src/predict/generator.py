import math
from collections.abc import Sequence

import torch

from src.anchor import PlaneAnchor, encode_anchors
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.predict.memory import estimate_memory, select_storage
from src.prepare.height import height_field


class GuidedDenoiser:
    def __init__(
        self, generator: "Generator", guidance: float, anchor_strength: float = 1.0
    ) -> None:
        self.generator = generator
        self.guidance = guidance
        self.anchor_strength = anchor_strength

    def __call__(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        height: torch.Tensor | None = None,
        coarse: torch.Tensor | None = None,
        corruption_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.generator.predict(
            current,
            time,
            latent,
            guidance=self.guidance,
            anchor_strength=self.anchor_strength,
            domain=domain,
            vf=vf,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
            **({"height": height} if height is not None else {}),
            **(
                {"coarse": coarse, "corruption_level": corruption_level}
                if coarse is not None
                else {}
            ),
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
        height_data: dict | None = None,
    ) -> None:
        self.model = model
        self.height_data = height_data
        self.num_domains = model.num_domains
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.use_amp = use_amp
        factor = getattr(model, "downsample_factor", None)
        if factor is None:
            if isinstance(model, Denoiser3D):
                raise AttributeError("Denoiser3D must expose downsample_factor.")
            factor = 1
        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
            raise ValueError("model.downsample_factor must be a positive integer.")
        self.default_margin = factor

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
        height: torch.Tensor | None = None,
        anchor_strength: float = 1.0,
        coarse: torch.Tensor | None = None,
        corruption_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor_strength == 0:
            anchor_image = anchor_mask = None
        elif anchor_mask is not None:
            anchor_mask = anchor_mask.float() * anchor_strength
        conditions = self.prepare_conditions(
            domain, vf, anchor_image, anchor_mask, height, coarse, corruption_level
        )
        if guidance == 1.0:
            return self.model(current, time, latent, **conditions)
        logits = self.compute_logits(
            current, time, latent, guidance=guidance, **conditions
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
        height: torch.Tensor | None = None,
        coarse: torch.Tensor | None = None,
        corruption_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditions = self.prepare_conditions(
            domain, vf, anchor_image, anchor_mask, height, coarse, corruption_level
        )
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
        height: torch.Tensor | None = None,
        coarse: torch.Tensor | None = None,
        corruption_level: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conditions = {"domain": domain}
        if coarse is not None:
            conditions["coarse"] = coarse
        if corruption_level is not None:
            conditions["corruption_level"] = corruption_level
        if height is not None:
            conditions["height"] = height
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
        height_origin: float = 0.0,
    ) -> torch.Tensor:
        size = self.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
        margin = self.default_margin if margin is None else margin
        if not isinstance(margin, int) or isinstance(margin, bool) or margin < 0:
            raise ValueError("margin must be a non-negative integer.")
        self.validate_anchor_strength(anchor_strength)
        self.validate_height((size,) * 3, domain, height_origin)
        select_storage(
            self.device.type,
            estimate_memory(size, self.num_phases, margin=margin, probabilities=True),
            self,
        )
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
            shifted_anchors = self.offset_anchors(anchors, size, margin)
            anchor = encode_anchors(
                shifted_anchors,
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=generation_size,
                device=self.device,
                dtype=initial_noise.dtype,
            )
        conditions = {"domain": self.prepare_domain(domain)}
        height = self.height_condition(
            (generation_size,) * 3, domain, height_origin, -margin
        )
        if height is not None:
            conditions["height"] = height
        if vf is not None:
            conditions["vf"] = vf
        if anchor is not None:
            conditions.update(anchor_image=anchor.image, anchor_mask=anchor.mask)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            clean = self.diffusion.sample(
                GuidedDenoiser(self, guidance, anchor_strength),
                initial_noise,
                self.latent_channels,
                conditions=conditions,
            )
        return self.crop_clean(clean, size, margin)

    def height_condition(self, shape, domain, origin, offset=0):
        if self.height_data is None:
            return None
        if not math.isfinite(origin) or origin < 0:
            raise ValueError(
                "height_origin must be finite and non-negative (source pixels)."
            )
        data = self.height_data
        domain = 0 if domain is None else domain
        axis = {"z": 0, "y": 1, "x": 2}[data["thickness_axis"]]
        spacing = data["crop_size"] / data["lo_res_size"]
        return height_field(
            shape,
            axis,
            origin + offset * spacing,
            spacing,
            data["height_extents"][domain],
            self.device,
        )

    def validate_height(self, shape, domain, origin):
        if self.height_data is None:
            return
        domain = 0 if domain is None and self.num_domains == 1 else domain
        if type(domain) is not int or not 0 <= domain < self.num_domains:
            raise ValueError("height conditioning requires a valid domain.")
        data = self.height_data
        axis = {"z": 0, "y": 1, "x": 2}[data["thickness_axis"]]
        extent = data["height_extents"][0 if domain is None else domain]
        length = shape[axis] * data["crop_size"] / data["lo_res_size"]
        if not math.isfinite(origin) or origin < 0 or origin + length > extent:
            raise ValueError(
                "height_origin places the volume outside the measured thickness."
            )

    @staticmethod
    def validate_anchor_strength(strength: float) -> None:
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not 0 <= strength <= 1
        ):
            raise ValueError("anchor_strength must be between zero and one.")

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
                4,
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
        height_origin: float = 0.0,
    ) -> torch.Tensor:
        clean = self._sample_clean(
            anchors=anchors,
            vf=vf,
            size=size,
            anchor_strength=anchor_strength,
            guidance=guidance,
            domain=domain,
            margin=margin,
            height_origin=height_origin,
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
        height_origin: float = 0.0,
    ) -> torch.Tensor:
        return (
            self.generate_probs(
                anchors=anchors,
                vf=vf,
                size=size,
                anchor_strength=anchor_strength,
                guidance=guidance,
                domain=domain,
                margin=margin,
                height_origin=height_origin,
            )
            .argmax(dim=0)
            .to(torch.uint8)
        )
