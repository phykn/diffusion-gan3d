import math
from collections.abc import Sequence
from dataclasses import replace
from functools import partial

import torch

from src.anchor import PlaneAnchor, anchor_origin, encode_anchors
from src.device import wide_dtype
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.phase import validate_num_phases
from src.predict.convert import labels_from_channels, owned_clean_to_probs_
from src.predict.memory import estimate_memory, select_storage
from src.prepare.height import height_field, resolve_extent
from src.prepare.profile import sample_profile


def profile_feasible(target, known, free, tolerance: float = 1e-4) -> bool:
    """Whether per-slice phase voxel targets fit fixed (known) and free voxels."""
    return not bool(
        ((target < known - tolerance) | (target > known + free + tolerance)).any()
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
        self.num_phases = validate_num_phases(num_phases)
        self.latent_channels = latent_channels
        self.use_amp = use_amp
        self.default_margin = model.downsample_factor

    def predict(
        self,
        current: torch.Tensor,
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        guidance: float = 1.0,
        anchor_strength: float = 1.0,
        **conditions,
    ) -> torch.Tensor:
        if anchor_strength == 0:
            conditions.update(anchor_image=None, anchor_mask=None)
        conditions = {
            name: value for name, value in conditions.items() if value is not None
        }
        partial_anchor = "anchor_mask" in conditions and 0.0 < anchor_strength < 1.0
        if guidance == 1.0 and not partial_anchor:
            return self.model(current, time, latent, domain=domain, **conditions)
        logits = self.model.apply_guidance_logits(
            current,
            time,
            latent,
            guidance,
            **({"anchor_strength": anchor_strength} if partial_anchor else {}),
            domain=domain,
            **conditions,
        )
        return Denoiser3D.decode(logits).to(current.dtype)

    def resolve_domain(self, domain: int | None) -> int:
        if domain is None:
            if self.num_domains != 1:
                raise ValueError("domain is required for a multi-domain model.")
            return 0
        if (
            not isinstance(domain, int)
            or isinstance(domain, bool)
            or not 0 <= domain < self.num_domains
        ):
            raise ValueError(
                f"domain must be an integer from 0 to {self.num_domains - 1}."
            )
        return domain

    def prepare_domain(self, domain: int | None) -> torch.Tensor:
        return torch.tensor(
            (self.resolve_domain(domain),),
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
            dtype=wide_dtype(self.device),
        )
        if values.shape != (self.num_phases,):
            raise ValueError(f"vf must have shape [{self.num_phases}].")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("vf values must be finite.")
        if bool((values < 0).any()):
            raise ValueError("vf values must be non-negative.")
        maximum = values.max()
        if maximum == 0:
            raise ValueError("vf sum must not be zero.")
        values = values / maximum
        return (values / values.sum()).to(torch.float32).unsqueeze(0)

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
        height_extent: float | None = None,
        vf_profile: dict | None = None,
        probabilities: bool = False,
    ) -> torch.Tensor:
        domain = self.resolve_domain(domain)
        size = self.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
        margin = self.default_margin if margin is None else margin
        if not isinstance(margin, int) or isinstance(margin, bool) or margin < 0:
            raise ValueError("margin must be a non-negative integer.")
        guidance = self.validate_guidance(guidance)
        anchor_strength = self.validate_anchor_strength(anchor_strength)
        height_extent = self.validate_height(
            (size,) * 3, domain, height_origin, height_extent
        )
        if vf_profile is None and vf is None and not anchors:
            vf_profile = self.default_profile(domain)
        self.validate_profile_request(
            vf_profile, size, domain, height_origin, height_extent, vf
        )
        select_storage(
            "cuda" if self.device.type == "cuda" else "cpu",
            estimate_memory(
                size, self.num_phases, margin=margin, probabilities=probabilities
            ),
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
            anchor = encode_anchors(
                self.offset_anchors(anchors, size, margin),
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=generation_size,
                device=self.device,
                dtype=initial_noise.dtype,
            )
        conditions = {
            "domain": self.prepare_domain(domain),
            "height": self.height_condition(
                (generation_size,) * 3, domain, height_origin, -margin, height_extent
            ),
            "vf": vf,
        }
        if vf_profile is not None:
            profile = self.profile_condition(
                vf_profile,
                generation_size,
                domain,
                height_origin,
                -margin,
                height_extent,
            )
            conditions.update(profile=profile, vf=profile.mean(-1))
            if anchor is not None and anchor_strength == 1:
                region = (..., *(slice(margin, margin + size),) * 3)
                mask = anchor.mask[region]
                known = ((anchor.image[region] + 1) * 0.5 * mask).sum((-1, -2))
                area = size * size
                target = profile[..., margin : margin + size] * area
                if not profile_feasible(target, known, area - mask.sum((-1, -2))):
                    raise ValueError(
                        "profile conflicts with fixed anchor phase fractions."
                    )
        if anchor is not None:
            conditions.update(anchor_image=anchor.image, anchor_mask=anchor.mask)
        clean = self.sample(initial_noise, conditions, guidance, anchor_strength)
        if margin:
            region = slice(margin, margin + size)
            clean = clean[:, :, region, region, region].clone()
        return clean

    def sample(self, noise, conditions, guidance, anchor_strength=1.0):
        """Run the full reverse diffusion from noise to clean phase values."""
        model = partial(
            self.predict, guidance=guidance, anchor_strength=anchor_strength
        )
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
        ):
            return self.diffusion.sample(
                model, noise, self.latent_channels, conditions=conditions
            )

    def height_condition(self, shape, domain, origin, offset=0, extent=None):
        """Height field over this model's output voxels, offset in those voxels."""
        if self.height_data is None:
            return None
        data = self.height_data
        spacing = data["crop_size"] / self.patch_size
        return height_field(
            shape,
            0,
            origin + offset * spacing,
            spacing,
            resolve_extent(data, self.resolve_domain(domain), extent),
            self.device,
        )

    def validate_height(self, shape, domain, origin, extent=None):
        """Check that an LR-voxel shape at origin fits the source image height."""
        if self.height_data is None:
            if extent is not None:
                raise ValueError("height_extent requires a height-conditioned model.")
            return
        data = self.height_data
        extent = resolve_extent(data, self.resolve_domain(domain), extent)
        length = shape[0] * data["crop_size"] / data["lo_res_size"]
        if not math.isfinite(origin) or origin < 0 or origin + length > extent:
            raise ValueError(
                "height_origin places the volume outside the source image height."
            )

        return extent

    def profile_condition(self, spec, depth, domain, origin, offset=0, extent=None):
        data = self.height_data
        spacing = data["crop_size"] / self.patch_size
        return sample_profile(
            spec,
            self.num_phases,
            depth,
            origin + offset * spacing,
            spacing,
            resolve_extent(data, self.resolve_domain(domain), extent),
            self.device,
        )

    def default_profile(self, domain):
        """Training-data depth profile; height alone flattens strong gradients."""
        if getattr(self.model, "profile_input", None) is None:
            return None
        profiles = self.height_data.get("depth_profiles") or {}
        return profiles.get(domain)

    def validate_profile_request(self, spec, depth, domain, origin, extent, vf):
        if spec is None:
            return
        if getattr(self.model, "profile_input", None) is None:
            raise ValueError("vf_profile requires a profile-conditioned LR model.")
        profile = self.profile_condition(spec, depth, domain, origin, extent=extent)
        supplied = self.prepare_vf(vf)
        if supplied is not None and not torch.allclose(
            supplied, profile.mean(-1), atol=1e-5, rtol=0
        ):
            raise ValueError(
                "vf must equal the profile mean over the final output interval."
            )

    @staticmethod
    def validate_anchor_strength(strength: float) -> float:
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not 0 <= strength <= 1
        ):
            raise ValueError("anchor_strength must be between zero and one.")
        return float(strength)

    @staticmethod
    def validate_guidance(guidance: float) -> float:
        if not math.isfinite(guidance):
            raise ValueError("guidance must be finite.")
        return float(guidance)

    @staticmethod
    def offset_anchors(
        anchors: Sequence[PlaneAnchor],
        size: int,
        margin: int,
    ) -> tuple[PlaneAnchor, ...]:
        anchors = tuple(anchors)
        for anchor in anchors:
            # A bool would pass the range check and shift into a valid index.
            if type(anchor.index) is not int or not 0 <= anchor.index < size:
                raise ValueError("anchor.index is outside the generated volume.")
            if not isinstance(anchor.image, torch.Tensor) or anchor.image.ndim not in {
                2,
                3,
                4,
            }:
                continue
            anchor_origin(anchor, anchor.image.shape[-2:], (size, size))
        if margin == 0:
            return anchors
        return tuple(
            replace(
                anchor,
                index=anchor.index + margin,
                position=None
                if anchor.position is None
                else (anchor.position[0] + margin, anchor.position[1] + margin),
            )
            for anchor in anchors
        )

    def generate_probs(self, **options) -> torch.Tensor:
        """Phase fractions (C,D,H,W); options are those of _sample_clean."""
        clean = self._sample_clean(probabilities=True, **options)
        return owned_clean_to_probs_(clean).squeeze(0).cpu()

    def generate(self, **options) -> torch.Tensor:
        """uint8 labels (D,H,W); options are those of _sample_clean."""
        clean = self._sample_clean(probabilities=False, **options)
        # Chunked argmax avoids a full int64 label volume on the model device.
        return labels_from_channels(clean.squeeze(0))
