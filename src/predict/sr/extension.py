import math
from itertools import product

import torch

from src.config.data import validate_sr_source
from src.predict.memory import base_bytes, ram_budget, require_memory
from src.predict.tiling.layout import parse_shape
from src.predict.volume_condition import fraction_channels, resolve_offset, volume_shape
from src.prepare.resize import downsample


@torch.inference_mode()
def coarsen_volume(high, scale, num_phases):
    """Average HR phase occupancy over scale^3 cells; shapes must align."""
    shape = volume_shape(high)
    low_shape = tuple(n // scale for n in shape)
    edge = max(1, 64 // scale)
    scratch = tuple(min(n, edge * scale) for n in shape)
    require_memory(
        4 * num_phases * math.prod(low_shape) + base_bytes(num_phases, scratch),
        ram_budget(),
        "coarse conversion RAM",
    )
    low = torch.empty(num_phases, *low_shape, dtype=torch.float32)
    for start in product(*(range(0, n, edge) for n in low_shape)):
        target = tuple(slice(s, min(s + edge, n)) for s, n in zip(start, low_shape))
        source = tuple(slice(p.start * scale, p.stop * scale) for p in target)
        block = high[(..., *source)]
        fractions = fraction_channels(block, num_phases)
        low[(slice(None), *target)] = downsample(
            fractions, tuple(p.stop - p.start for p in target)
        )[0]
    return low


# no_grad, not inference_mode: inference tensors recompile the fused norm.
@torch.no_grad()
def extend_hr(
    lr,
    sr,
    base,
    shape,
    *,
    low=None,
    base_offset=(0, 0, 0),
    anchors=(),
    anchor_strength=None,
    vf=None,
    domain=None,
    seed=None,
    lr_overlap=None,
    tile_size=None,
    overlap=None,
    margin=None,
    guidance=None,
    sr_guidance=1.0,
    storage="auto",
    height_origin=0.0,
    height_extent=None,
    vf_profile=None,
    base_height_origin=None,
    progress=False,
    probabilities=False,
):
    validate_sr_source(sr.config["data"], lr.data)
    if not sr.scale_factor.is_integer():
        raise ValueError("HR extension requires an integer SR scale.")
    scale = int(sr.scale_factor)
    shape = parse_shape(shape)
    base_shape = volume_shape(base)
    if base_shape is None:
        raise ValueError("HR extension requires base.")
    position = resolve_offset(base_offset, shape, base_shape)
    if any(n % scale for n in (*shape, *base_shape, *position)):
        raise ValueError(
            "HR shape, base shape and base_offset must align with the SR scale."
        )
    low_shape = tuple(n // scale for n in shape)
    old_shape = tuple(n // scale for n in base_shape)
    if any(n < lr.input_size for n in low_shape):
        raise ValueError("extended LR shape must be at least the LR model grid.")
    if sr.config["conditioning"]["height_enabled"]:
        expected = height_origin + position[0] * sr.crop_size / sr.hi_res_size
        if base_height_origin is None or not math.isclose(base_height_origin, expected):
            raise ValueError(
                "base_height_origin must match its position in the extended height field."
            )
    if low is not None and volume_shape(low) != old_shape:
        raise ValueError("original LR shape must match the existing HR base.")
    if tile_size is None:
        tile_size = sr.hi_res_size
    # Check the HR tiling and memory before the LR volume is generated.
    sr.check_memory(
        sr.estimate_memory(
            low_shape,
            tile_size=tile_size,
            overlap=overlap,
            margin=margin,
            label_input=False,
            input_element_size=4,
            probabilities=probabilities,
        )
    )
    if low is None:
        low = coarsen_volume(base, scale, sr.num_phases)
    expanded = lr.generate_probs(
        shape=low_shape,
        base=low,
        base_offset=tuple(n // scale for n in position),
        preserve_base=True,
        anchors=anchors,
        anchor_strength=anchor_strength,
        vf=vf,
        vf_profile=vf_profile,
        domain=domain,
        seed=seed,
        overlap=lr_overlap,
        guidance=guidance,
        storage=storage,
        height_origin=height_origin,
        height_extent=height_extent,
        progress=progress,
    )
    refine = sr.refine_probs if probabilities else sr.refine
    high = refine(
        expanded,
        domain=domain,
        seed=seed,
        tile_size=tile_size,
        overlap=overlap,
        margin=margin,
        guidance=sr_guidance,
        height_origin=height_origin,
        height_extent=height_extent,
        base=base,
        base_offset=position,
    )
    return expanded, high
