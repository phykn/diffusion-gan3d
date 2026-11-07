import math

import pytest
import torch

from src.sampling.plan import make_plan, tile_window


def test_one_tile_when_the_region_fits():
    plan = make_plan((16, 16, 16), 16, margin=8, overlap=4)
    assert plan.shape == (32, 32, 32)
    assert plan.tile_count == 1
    assert plan.tiles[0].shape == (32, 32, 32)


@pytest.mark.parametrize("size", [17, 40, 64])
def test_tiles_cover_the_region_once(size):
    plan = make_plan((size, 16, 16), 16, margin=2, overlap=3)
    owner = torch.zeros(plan.shape, dtype=torch.long)
    weight = torch.zeros(plan.shape)
    for tile in plan.tiles:
        owner[tile.target] += 1
        weight[tile.source] += tile_window(tile, plan.overlap)[0, 0]
        assert tile.shape == plan.tile_edge
    assert bool((owner == 1).all())
    assert bool((weight > 0).all())


def test_bad_overlap():
    with pytest.raises(ValueError, match="overlap"):
        make_plan((64, 16, 16), 16, margin=0, overlap=8)


def test_counting_a_huge_plan_builds_no_tiles():
    # Estimates and web limits count tiles before anything is sampled.
    plan = make_plan((512, 512, 512), 16, 8, 15)
    assert plan.tile_count == math.prod(plan.grid) > 10**6
    assert "tiles" not in vars(plan)
