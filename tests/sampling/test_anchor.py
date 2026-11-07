from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.api import LowResolutionAPI, PlaneAnchor
from src.sampling.anchor import AnchorGuide, TileAnchors, encode, place_anchors
from src.sampling.plan import make_plan

REFERENCE = torch.load(
    Path(__file__).parents[1] / "fixtures" / "parity" / "anchor.pt", weights_only=False
)


def anchor_model(weights: dict, values, time, latent):
    """The stand-in denoiser v1 guided to write fixtures/parity/anchor.pt."""
    mixed = torch.einsum("kc,bcdhw->bkdhw", weights["mix"], values)
    shifted = values.roll(1, dims=3) * weights["shift"][None, :, None, None, None]
    bias = weights["bias"][None, :, None, None, None] * (time.float() + 1) / 10
    return torch.tanh(mixed + shifted + bias + 0.01 * latent.sum())


def reference_anchors() -> list[PlaneAnchor]:
    """An xy anchor and an xz anchor that meet on the line z = 2, y = 3."""
    return [PlaneAnchor(**item) for item in REFERENCE["anchors"]]


def test_anchor_parity():
    shape = REFERENCE["shape"]
    image, mask = encode(
        place_anchors(reference_anchors(), shape, 3), shape, check=True
    )
    torch.testing.assert_close(image, REFERENCE["image"], atol=1e-6, rtol=0)
    assert torch.equal(mask, REFERENCE["mask"])
    weights = REFERENCE["weights"]
    guide = AnchorGuide(image, mask, REFERENCE["strength"])
    pred, step = guide(
        lambda values, time, latent, guidance, **_: anchor_model(
            weights, values, time, latent
        ),
        REFERENCE["state"],
        REFERENCE["time"],
        REFERENCE["latent"],
        1.0,
        {},
    )
    torch.testing.assert_close(pred, REFERENCE["pred"], atol=1e-5, rtol=0)
    torch.testing.assert_close(step, REFERENCE["step"], atol=1e-5, rtol=0)


def test_conflicting_intersections_are_rejected():
    conflict = REFERENCE["conflict"]
    assert conflict["error"] == "anchor planes contain conflicting intersections."
    anchors = reference_anchors()
    anchors[1] = replace(anchors[1], image=conflict["fractions"])
    shape = REFERENCE["shape"]
    with pytest.raises(ValueError, match="conflicting intersections"):
        encode(place_anchors(anchors, shape, 3), shape, check=True)
    # The tiles check the same intersection before sampling.
    plan = make_plan(shape, 4, margin=2, overlap=1)
    with pytest.raises(ValueError, match="conflicting intersections"):
        TileAnchors(anchors, plan, 3, 1.0)


def test_each_tile_gets_its_part_of_the_anchors():
    anchors = reference_anchors()
    plan = make_plan(REFERENCE["shape"], 4, margin=2, overlap=1)
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
def tiny_api(tiny_configs, random_export) -> LowResolutionAPI:
    return LowResolutionAPI(random_export(tiny_configs["lr"], "tiny_lr"), "cpu")


def test_zero_strength_matches_no_anchor(tiny_api):
    labels = torch.randint(tiny_api.num_phases, (16, 16), generator=torch.Generator())
    anchor = PlaneAnchor(labels, 0, 5)
    plain = tiny_api.generate(size=24, seed=1)
    anchored = tiny_api.generate([anchor], size=24, seed=1, anchor_strength=0.0)
    assert torch.equal(plain, anchored)
    guided = tiny_api.generate([anchor], size=24, seed=1, anchor_strength=1.0)
    assert not torch.equal(plain, guided)
