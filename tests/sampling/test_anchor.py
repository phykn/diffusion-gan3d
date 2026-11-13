from dataclasses import replace

import pytest
import torch

from src.api import LowResolutionAPI, PlaneAnchor
from src.sampling.anchor import TileAnchors, encode, place_anchors
from src.sampling.plan import make_plan

SHAPE = (6, 8, 10)


def reference_anchors():
    image = torch.zeros((4, 4), dtype=torch.long)
    return [PlaneAnchor(image, 0, 2, (2, 2)), PlaneAnchor(image, 1, 3, (0, 2))]


def test_conflicting_intersections_are_rejected():
    anchors = reference_anchors()
    anchors[1] = replace(anchors[1], image=torch.ones_like(anchors[1].image))
    shape = SHAPE
    with pytest.raises(ValueError, match="conflicting intersections"):
        encode(place_anchors(anchors, shape, 3), shape, check=True)
    # The tiles check the same intersection before sampling.
    plan = make_plan(shape, 4, margin=2, overlap=1)
    with pytest.raises(ValueError, match="conflicting intersections"):
        TileAnchors(anchors, plan, 3, 1.0)


def test_each_tile_gets_its_part_of_the_anchors():
    anchors = reference_anchors()
    plan = make_plan(SHAPE, 4, margin=2, overlap=1)
    assert plan.tile_count > 1
    whole_image, whole_mask = encode(
        place_anchors(anchors, plan.output_shape, 3, plan.margin), plan.shape
    )
    tiles = TileAnchors(anchors, plan, 3, 1.0)
    for index, tile in enumerate(plan.tiles):
        region = (slice(None), slice(None), *tile.source)
        guide = tiles(index, tile)
        if guide is None:
            assert not whole_mask[region].any()
            continue
        assert torch.equal(guide.mask, whole_mask[region])
        torch.testing.assert_close(guide.image, whole_image[region])


@pytest.mark.parametrize(
    ("anchor", "error", "match"),
    [
        ("not an anchor", TypeError, "PlaneAnchor"),
        (PlaneAnchor(torch.zeros(4, 4, dtype=torch.long), True, 0), ValueError, "axis"),
        (PlaneAnchor(torch.zeros(4, 4, dtype=torch.long), 0, 6), ValueError, "index"),
        (
            PlaneAnchor(torch.zeros(4, 4, dtype=torch.long), 0, 0, (5, 0)),
            ValueError,
            "outside the plane",
        ),
        (PlaneAnchor(torch.zeros(9, 4, dtype=torch.long), 0, 0), ValueError, "fit"),
        (PlaneAnchor(torch.full((3, 4, 4), 0.5), 0, 0), ValueError, "sum to one"),
    ],
)
def test_invalid_anchors_are_rejected(anchor, error, match):
    with pytest.raises(error, match=match):
        place_anchors([anchor], (6, 8, 10), 3)


@pytest.fixture(scope="module")
def tiny_api(tiny_configs, random_export):
    return LowResolutionAPI(random_export(tiny_configs["lr"], "tiny_lr"), "cpu")


def test_zero_strength_matches_no_anchor(tiny_api):
    labels = torch.randint(tiny_api.num_phases, (16, 16), generator=torch.Generator())
    anchor = PlaneAnchor(labels, 0, 5)
    plain = tiny_api.generate(shape=24, seed=1)
    anchored = tiny_api.generate([anchor], shape=24, seed=1, anchor_strength=0.0)
    assert tiny_api.estimate_memory(shape=24, anchors=[anchor], anchor_strength=0) == (
        tiny_api.estimate_memory(shape=24)
    )
    assert torch.equal(plain, anchored)
    guided = tiny_api.generate([anchor], shape=24, seed=1, anchor_strength=1.0)
    assert not torch.equal(plain, guided)
