import math

import torch
import torch.nn.functional as F

from src.data.height import build_heights, check_extent
from src.data.profile import integrate_profile, validate_profile
from src.data.resize import upsample_region
from src.device import wide_dtype

HEIGHT_TOLERANCE = 1e-6
VF_TOLERANCE = 1e-5


class HeightRange:
    def __init__(self, config, plan, origin, extent):
        self.enabled = config.conditioning.height_enabled
        self.pixels_per_cell = config.data.crop_size / config.patch_size
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
        output_height = plan.output_shape[0] * self.pixels_per_cell
        if (
            not math.isfinite(self.origin)
            or self.origin < 0
            or self.origin + output_height > self.extent + HEIGHT_TOLERANCE
        ):
            raise ValueError(
                "height_origin places the volume outside the source image height."
            )

    def locate_tile(self, tile):
        cells_below_origin = tile.source[0].start - self.margin
        return self.origin + cells_below_origin * self.pixels_per_cell

    def build_heights(self, tile, device):
        return build_heights(
            tile.shape,
            self.locate_tile(tile),
            self.pixels_per_cell,
            self.extent,
            device,
        )


class LRConditions:
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
        self.vf = _normalize_vf(vf, self.num_phases, device)
        if self.profile is not None and self.vf is not None:
            profile = self._integrate_profile(plan.output_shape[0], self.height.origin)
            if not torch.allclose(self.vf, profile.mean(-1), atol=VF_TOLERANCE, rtol=0):
                raise ValueError(
                    "vf must equal the profile mean over the final output interval."
                )

    def __call__(self, tile):
        conditions = {"vf": self.vf}
        if self.height.enabled:
            conditions["height"] = self.height.build_heights(tile, self.device)
            if self.profile is not None:
                profile = self._integrate_profile(
                    tile.shape[0], self.height.locate_tile(tile)
                )
                conditions.update(profile=profile, vf=profile.mean(-1))
        return conditions

    @property
    def has_measured_condition(self):
        return self.vf is not None or self.profile is not None

    def _integrate_profile(self, depth, origin):
        return integrate_profile(
            *self.profile,
            depth,
            origin,
            self.height.pixels_per_cell,
            self.height.extent,
            self.device,
        )


class SRConditions:
    # `low` is the LR fractions [K, d, h, w] of the output.
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
        expected = tuple(size // self.scale for size in plan.output_shape)
        if low.ndim != 4 or tuple(low.shape[1:]) != expected:
            raise ValueError(f"LR fractions must have shape [C, {expected}].")
        margin_cells = plan.margin // self.scale
        low = low.detach().to("cpu", torch.float32)[None]
        # Outside the output the coarse input repeats its faces.
        self.low = (
            F.pad(low, (margin_cells,) * 6, mode="replicate") if margin_cells else low
        )
        self.device = device
        self.height = HeightRange(config, plan, height_origin, height_extent)

    def __call__(self, tile):
        start = tuple(region.start for region in tile.source)
        coarse = upsample_region(self.low, start, tile.shape, self.scale)
        conditions = {"coarse": coarse.to(self.device, torch.float32)}
        if self.height.enabled:
            conditions["height"] = self.height.build_heights(tile, self.device)
        return conditions


def _normalize_vf(vf, num_phases, device):
    if vf is None:
        return None
    values = torch.as_tensor(vf, dtype=wide_dtype(torch.device(device)))
    if values.shape != (num_phases,):
        raise ValueError(f"vf must have shape [{num_phases}].")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("vf values must be finite.")
    if bool((values < 0).any()):
        raise ValueError("vf values must be non-negative.")
    largest = values.max()
    if largest == 0:
        raise ValueError("vf sum must not be zero.")
    values = values / largest
    return (values / values.sum()).to(device=device, dtype=torch.float32)[None]
