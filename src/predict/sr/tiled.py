from dataclasses import replace

import torch.nn.functional as F

from src.predict.tiling.layout import make_plan, make_tiles
from src.predict.tiling.sampler import TileSampler
from src.predict.volume_condition import prepare_base
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
    plan = replace(
        make_plan(generation_shape, tile_size + 2 * margin, overlap + margin),
        generation_shape=generation_shape,
        margin=margin,
    )
    tiles = make_tiles(plan)
    known = prepare_base(
        base, api.num_phases, shape, margin, base_offset, 0, base is not None
    )

    def conditions(tile, index):
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

    return TileSampler(api.generator).sample(
        tiles,
        plan,
        base=known,
        vf=None,
        domain=domain,
        probabilities=output_kind == "probabilities",
        guidance=guidance,
        tile_conditions=conditions,
    )
