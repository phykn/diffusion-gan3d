import math
from itertools import pairwise

import torch.nn.functional as F

from src.predict.base import prepare_base
from src.predict.convert import labels_from_channels
from src.predict.tiling.layout import TilePlan, axis_starts, make_tiles
from src.predict.tiling.sampler import TiledGenerator
from src.predict.tiling.state import collect_probabilities
from src.prepare.resize import coarse_region, resize_phases


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
    base=None,
    base_offset=None,
    height_extent=None,
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
    known = prepare_base(
        base, api.num_phases, shape, margin, base_offset, 0, base is not None
    )
    current, next_state = sampler.make_states(plan, "cpu")
    sampler.fill_noise(current, tiles)

    def conditions(tile):
        origin = tuple(part.start - margin for part in tile.source)
        expanded = tuple(part.stop - part.start for part in tile.source)
        if len(tiles) > 1:
            values = coarse_region(coarse, origin, expanded, int(api.scale_factor))
        else:
            # A single block also supports fractional scale and arbitrary margin.
            values = F.pad(
                resize_phases(coarse, shape), (margin,) * 6, mode="replicate"
            )
        values = values.to(api.device)
        result = {"coarse": values, "corruption_level": values.new_zeros(1)}
        height = api._height(
            expanded, origin, int(domain.item()), height_origin, height_extent
        )
        if height is not None:
            result["height"] = height
        return result

    current = sampler.sample(
        current,
        next_state,
        tiles,
        plan,
        base=known,
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
    return collect_probabilities(current, tiles, shape, margin, api.num_phases)
