import math

import torch
import torch.nn.functional as F

from src.data.height import build_heights, check_extent
from src.data.profile import integrate_profile, validate_profile
from src.data.resize import upsample_region
from src.device import wide_dtype
from src.sampling.plan import make_plan, parse_shape

HEIGHT_TOLERANCE = 1e-6


def make_height_plan(
    config, output_shape, patch, margin, overlap, origin=0.0, extent=None, step=1
):
    """make_plan with z margins that stay inside the source height."""
    depth = parse_shape(output_shape, "output shape")[0]
    z_margins = HeightRange(config, origin, extent).z_margins(depth, margin, step)
    return make_plan(output_shape, patch, margin, overlap, z_margins)


class HeightRange:
    """Where an output lies in the source height, in source pixels."""

    def __init__(self, config, origin, extent):
        self.enabled = config.conditioning.height_enabled
        self.pixels_per_cell = config.data.crop_size / config.patch_size
        self.top_margin = 0
        if not self.enabled:
            if extent is not None or origin != 0:
                raise ValueError(
                    "height_origin and height_extent require a height-conditioned "
                    "model."
                )
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

    def z_margins(self, depth, margin, step=1):
        """Top and bottom margins of a depth-cell output, cut to whole steps of
        cells inside the source: training never sees a height past a z face."""
        if not self.enabled:
            return margin, margin
        above, below = self._cells_around(depth)
        return tuple(
            min(margin, math.floor(cells + HEIGHT_TOLERANCE) // step * step)
            for cells in (above, below)
        )

    def place(self, plan):
        """Check that the plan's z margins stay inside the source height."""
        self.top_margin = plan.z_margins[0]
        if not self.enabled:
            return
        above, below = self._cells_around(plan.output_shape[0])
        top, bottom = plan.z_margins
        if top > above + HEIGHT_TOLERANCE or bottom > below + HEIGHT_TOLERANCE:
            raise ValueError("z margins reach past the source image height.")

    def _cells_around(self, depth):
        """Cells of source height above and below a depth-cell output."""
        output_height = depth * self.pixels_per_cell
        if (
            not math.isfinite(self.origin)
            or self.origin < 0
            or self.origin + output_height > self.extent + HEIGHT_TOLERANCE
        ):
            raise ValueError(
                "height_origin places the volume outside the source image height."
            )
        below = self.extent - self.origin - output_height
        return self.origin / self.pixels_per_cell, below / self.pixels_per_cell

    def tile_origin(self, tile):
        """Source row of the tile's top face."""
        cells_from_output_top = tile.source[0].start - self.top_margin
        return self.origin + cells_from_output_top * self.pixels_per_cell

    def build_heights(self, tile, device):
        return build_heights(
            tile.shape,
            self.tile_origin(tile),
            self.pixels_per_cell,
            self.extent,
            device,
        )


class LRConditions:
    """Per tile: vf, and for a height model the height field and the depth
    profile over the tile's rows (whose mean then replaces vf)."""

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
        num_phases = config.data.num_phases
        self.height = HeightRange(config, height_origin, height_extent)
        self.height.place(plan)
        if vf_profile is None and vf is None and config.profile_enabled:
            # Height alone flattens strong depth gradients; use the training one.
            vf_profile = config.data.depth_profile
        self.profile = None
        if vf_profile is not None:
            if not config.profile_enabled:
                raise ValueError("vf_profile requires a profile-conditioned LR model.")
            self.profile = (
                *validate_profile(vf_profile, num_phases),
                vf_profile["interpolation"],
            )
        self.vf = _normalize_vf(vf, num_phases, device)
        if self.profile is not None and self.vf is not None:
            whole = self._integrate_profile(plan.output_shape[0], self.height.origin)
            if not torch.allclose(self.vf, whole.mean(-1), atol=1e-5, rtol=0):
                raise ValueError(
                    "vf must equal the profile mean over the final output interval."
                )

    def __call__(self, tile):
        conditions = {"vf": self.vf}
        if self.height.enabled:
            conditions["height"] = self.height.build_heights(tile, self.device)
            if self.profile is not None:
                profile = self._integrate_profile(
                    tile.shape[0], self.height.tile_origin(tile)
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
    """Per tile: the LR fractions low [K, d, h, w] of the output enlarged to
    the tile, and for a height model the height field."""

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
        self.device = device
        self.scale = config.scale_factor
        expected = tuple(size // self.scale for size in plan.output_shape)
        if low.ndim != 4 or tuple(low.shape[1:]) != expected:
            raise ValueError(f"LR fractions must have shape [C, {expected}].")
        side = plan.margin // self.scale
        top, bottom = (cells // self.scale for cells in plan.z_margins)
        low = low.detach().to("cpu", torch.float32)[None]
        # Outside the output the coarse input repeats its faces.
        padding = (side, side, side, side, top, bottom)
        self.low = F.pad(low, padding, mode="replicate")
        self.height = HeightRange(config, height_origin, height_extent)
        self.height.place(plan)

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
    total = values.sum()
    if total == 0:
        raise ValueError("vf sum must not be zero.")
    return (values / total).to(device=device, dtype=torch.float32)[None]
