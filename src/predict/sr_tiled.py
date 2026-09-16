"""SR conditions on the shared, globally synchronized diffusion sampler."""

import math
from itertools import pairwise

import torch

from src.predict.convert import labels_from_channels, owned_clean_to_probs_
from src.predict.tile import TilePlan, axis_starts, make_tiles
from src.predict.tiled import TiledGenerator
from src.prepare.resize import coarse_region


def refine_tiled(
    api,
    coarse,
    shape,
    tile_size,
    overlap,
    margin,
    domain,
    height_origin,
    guidance,
    output_kind,
):
    # Include global context in the shared state. Each tile keeps its HR margin;
    # overlapping clean predictions are tapered before one posterior update.
    generation_shape = tuple(n + 2 * margin for n in shape)
    expanded_size = tile_size + 2 * margin
    stride = tile_size - 2 * overlap
    lengths = tuple(min(expanded_size, n) for n in generation_shape)
    starts = tuple(
        axis_starts(n, length, stride) for n, length in zip(generation_shape, lengths)
    )
    grid = tuple(len(axis) for axis in starts)
    plan = TilePlan(
        shape=generation_shape,
        tile_size=expanded_size,
        overlap=overlap + margin,
        stride=stride,
        grid=grid,
        tile_count=math.prod(grid),
        seams=tuple(
            tuple((left + length + right) // 2 for left, right in pairwise(axis))
            for axis, length in zip(starts, lengths)
        ),
        generation_shape=generation_shape,
        margin=margin,
    )
    tiles = make_tiles(plan)
    sampler = TiledGenerator(api.generator)
    current, next_state = sampler.make_states(plan, "cpu")
    sampler.fill_noise(current, tiles)

    def conditions(tile):
        origin = tuple(part.start - margin for part in tile.source)
        expanded = tuple(part.stop - part.start for part in tile.source)
        values = coarse_region(coarse, origin, expanded, int(api.scale_factor)).to(
            api.device
        )
        result = {"coarse": values, "corruption_level": values.new_zeros(1)}
        height = api._height(expanded, origin, int(domain.item()), height_origin)
        if height is not None:
            result["height"] = height
        return result

    current = sampler.sample(
        current,
        next_state,
        tiles,
        plan,
        base=None,
        vf=None,
        domain=domain,
        labels=None,
        progress=False,
        guidance=guidance,
        tile_conditions=conditions,
    )
    region = tuple(slice(margin, margin + n) for n in shape)
    if output_kind == "labels":
        return labels_from_channels(current.read(region).squeeze(0))
    output = torch.empty(api.num_phases, *shape, dtype=torch.float32)
    for tile in tiles:
        source = tuple(
            slice(max(part.start, margin), min(part.stop, margin + n))
            for part, n in zip(tile.target, shape)
        )
        if any(part.start >= part.stop for part in source):
            continue
        target = tuple(
            slice(part.start - margin, part.stop - margin) for part in source
        )
        values = owned_clean_to_probs_(current.read(source).float()).squeeze(0)
        output[(slice(None), *target)].copy_(values)
    return output
