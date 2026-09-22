import math
from itertools import product

import psutil
import torch

from src.config.data import validate_sr_source
from src.predict.base import phase_fractions, resolve_offset, volume_shape
from src.predict.memory import require_memory
from src.predict.tiling.layout import parse_shape
from src.prepare.resize import downsample


@torch.inference_mode()
def coarsen_volume(high, scale, num_phases):
    shape = volume_shape(high)
    if type(scale) is not int or scale < 1 or any(n % scale for n in shape):
        raise ValueError("HR base shape must align with an integer SR scale.")
    low_shape = tuple(n // scale for n in shape)
    edge = max(1, 64 // scale)
    scratch = math.prod(tuple(min(n, edge * scale) for n in shape))
    require_memory(
        4 * num_phases * math.prod(low_shape) + (12 * num_phases + 24) * scratch,
        int(psutil.virtual_memory().available * 0.8),
        "coarse conversion RAM",
    )
    low = torch.empty(num_phases, *low_shape, dtype=torch.float32)
    for start in product(*(range(0, n, edge) for n in low_shape)):
        target = tuple(slice(s, min(s + edge, n)) for s, n in zip(start, low_shape))
        source = tuple(slice(p.start * scale, p.stop * scale) for p in target)
        block = high[(..., *source)]
        fractions = phase_fractions(block, num_phases)
        low[(slice(None), *target)] = downsample(
            fractions, tuple(p.stop - p.start for p in target)
        )[0]
    return low


@torch.inference_mode()
def extend_hr(
    lr,
    sr,
    base,
    shape,
    *,
    low=None,
    base_offset=(0, 0, 0),
    anchors=(),
    vf=None,
    domain=None,
    seed=0,
    lr_overlap=None,
    tile_size=None,
    overlap=8,
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
    position, _ = resolve_offset(base_offset, shape, base_shape)
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
    if low is None:
        low = coarsen_volume(base, scale, sr.num_phases)
    expanded = lr.generate_probs(
        shape=low_shape,
        base=low,
        base_offset=tuple(n // scale for n in position),
        preserve_base=True,
        anchors=anchors,
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
    refine = sr.predict_probs if probabilities else sr.super_resolve
    high = refine(
        expanded,
        domain=domain,
        seed=seed,
        tile_size=sr.hi_res_size if tile_size is None else tile_size,
        overlap=overlap,
        margin=margin,
        guidance=sr_guidance,
        height_origin=height_origin,
        height_extent=height_extent,
        base=base,
        base_offset=position,
    )
    return expanded, high
