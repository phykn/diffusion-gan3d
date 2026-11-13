import math

import torch
import torch.nn.functional as F

from src.data.crops import coarse_region
from src.data.height import (
    check_extent,
    grid_spacing,
    height_field,
    integrate_profile,
    validate_profile,
)
from src.device import wide_dtype


def prepare_vf(vf, num_phases, device):
    if vf is None:
        return None
    values = torch.as_tensor(vf, dtype=wide_dtype(torch.device(device)))
    if values.shape != (num_phases,):
        raise ValueError(f"vf must have shape [{num_phases}].")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("vf values must be finite.")
    if bool((values < 0).any()):
        raise ValueError("vf values must be non-negative.")
    maximum = values.max()
    if maximum == 0:
        raise ValueError("vf sum must not be zero.")
    values = values / maximum
    return (values / values.sum()).to(device=device, dtype=torch.float32)[None]


class HeightRange:
    """Source-pixel height coordinates of a generation along z."""

    def __init__(self, config, plan, origin, extent):
        self.enabled = config.conditioning.height_enabled
        self.spacing = grid_spacing(config.data.crop_size, config.patch_size)
        self.margin = plan.margin
        if not self.enabled:
            if extent is not None:
                raise ValueError("height_extent requires a height-conditioned model.")
            self.origin, self.extent = 0.0, None
            return
        if extent is None:
            extent = config.data.height_extent
        if extent is None:
            raise ValueError(
                "height_extent is required when source images have different heights."
            )
        self.extent = check_extent(extent)
        if isinstance(origin, bool) or not isinstance(origin, (int, float)):
            raise ValueError("height_origin must be a number (source pixels).")
        self.origin = float(origin)
        # The output depth at this origin must lie inside the source height.
        length = plan.output_shape[0] * self.spacing
        if (
            not math.isfinite(self.origin)
            or self.origin < 0
            or self.origin + length > self.extent + 1e-6
        ):
            raise ValueError(
                "height_origin places the volume outside the source image height."
            )

    def tile_origin(self, tile):
        return self.origin + (tile.source[0].start - self.margin) * self.spacing

    def field(self, tile, device):
        return height_field(
            tile.shape, self.tile_origin(tile), self.spacing, self.extent, device
        )


class LRConditions:
    """vf, height and depth profile of each LR tile."""

    def __init__(
        self,
        config,
        plan,
        device,
        *,
        vf=None,
        vf_profile=None,
        height_origin=0.0,
        height_extent=None,
    ):
        self.device = device
        self.num_phases = config.data.num_phases
        self.height = HeightRange(config, plan, height_origin, height_extent)
        profile_enabled = config.profile_enabled
        if vf_profile is None and vf is None and profile_enabled:
            # Height alone flattens strong depth gradients; use the training one.
            vf_profile = config.data.depth_profile
        self.profile = None
        if vf_profile is not None:
            if not profile_enabled:
                raise ValueError("vf_profile requires a profile-conditioned LR model.")
            self.profile = (
                *validate_profile(vf_profile, self.num_phases),
                vf_profile["interpolation"],
            )
        self.vf = prepare_vf(vf, self.num_phases, device)
        if self.profile is not None and self.vf is not None:
            profile = self._profile(plan.output_shape[0], self.height.origin)
            if not torch.allclose(self.vf, profile.mean(-1), atol=1e-5, rtol=0):
                raise ValueError(
                    "vf must equal the profile mean over the final output interval."
                )

    @property
    def guided(self):
        """Whether CFG has a condition to drop."""
        return self.vf is not None or self.profile is not None

    def _profile(self, depth, origin):
        return integrate_profile(
            *self.profile,
            depth,
            origin,
            self.height.spacing,
            self.height.extent,
            self.device,
        )

    def __call__(self, index, tile):
        result = {"vf": self.vf}
        if self.height.enabled:
            result["height"] = self.height.field(tile, self.device)
            if self.profile is not None:
                profile = self._profile(tile.shape[0], self.height.tile_origin(tile))
                result.update(profile=profile, vf=profile.mean(-1))
        return result


class SRConditions:
    """Coarse LR fractions (with one LR cell of halo) and height of each tile.

    low is the LR fractions [K, d, h, w] of the output; outside the output the
    coarse input repeats its faces.
    """

    def __init__(
        self,
        config,
        plan,
        low,
        device,
        *,
        height_origin=0.0,
        height_extent=None,
    ):
        self.scale = config.scale_factor
        expected = tuple(n // self.scale for n in plan.output_shape)
        if low.ndim != 4 or tuple(low.shape[1:]) != expected:
            raise ValueError(f"LR fractions must have shape [C, {expected}].")
        pad = plan.margin // self.scale
        low = low.detach().to("cpu", torch.float32)[None]
        self.low = F.pad(low, (pad,) * 6, mode="replicate") if pad else low
        self.device = device
        self.height = HeightRange(config, plan, height_origin, height_extent)

    def __call__(self, index, tile):
        start = tuple(region.start for region in tile.source)
        coarse = coarse_region(self.low, start, tile.shape, self.scale)
        result = {"coarse": coarse.to(self.device, torch.float32)}
        if self.height.enabled:
            result["height"] = self.height.field(tile, self.device)
        return result
