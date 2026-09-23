from dataclasses import replace

import pytest

from src.predict.tiling.layout import make_plan, make_tiles, output_plan


def test_snapped_starts_and_owned_regions_share_generation_geometry():
    plan = make_plan((13, 8, 8), 8, 2)
    assert plan.starts == ((0, 4, 5), (0,), (0,))
    assert plan.seams == ((6, 8), (), ())
    tiles = make_tiles(plan)
    assert [tile.source[0] for tile in tiles] == [
        slice(0, 8), slice(4, 12), slice(5, 13)
    ]
    assert [tile.target[0] for tile in tiles] == [
        slice(0, 6), slice(6, 8), slice(8, 13)
    ]


def test_output_statistics_retain_padded_grid_but_cannot_generate_tiles():
    plan = replace(
        make_plan((16, 16, 16), 8, 2), generation_shape=(16, 16, 16), margin=2
    )
    stats = output_plan(plan, (12, 12, 12))
    assert stats.starts is plan.starts
    assert stats.grid == (3, 3, 3)
    assert stats.seams == ((4, 8), (4, 8), (4, 8))
    with pytest.raises(ValueError, match="output-space statistics"):
        make_tiles(stats)
