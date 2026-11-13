import pytest
import torch

import src.api
from src.api import LowResolutionAPI
from src.data.height import height_field, integrate_profile, validate_profile
from src.sampling.conditions import LRConditions


@pytest.fixture(scope="module")
def tiny_api(tiny_configs, random_export):
    return LowResolutionAPI(random_export(tiny_configs["lr"], "tiny_lr"), "cpu")


@pytest.fixture(scope="module")
def height_api(tiny_height_configs, random_export):
    config = tiny_height_configs["lr"]
    assert config.conditioning.height_enabled and config.profile_enabled
    return LowResolutionAPI(random_export(config, "height_lr"), "cpu")


def test_scalar_and_tuple_shape_give_the_same_volume(tiny_api):
    by_size = tiny_api.generate(shape=24, seed=3)
    by_shape = tiny_api.generate(shape=(24, 24, 24), seed=3)
    assert by_size.shape == (24, 24, 24)
    assert torch.equal(by_size, by_shape)


def test_blocks_give_the_tiled_size(tiny_api):
    # The default overlap is half the tiny patch; blocks need less.
    patch, overlap = tiny_api.input_size, 4
    edge = patch + (2 - 1) * (patch - 2 * overlap)
    volume = tiny_api.generate(blocks=(2, 2, 2), overlap=overlap, seed=0)
    assert volume.shape == (edge,) * 3
    estimate = tiny_api.estimate_memory(blocks=(2, 2, 2), overlap=overlap)
    assert estimate.shape == (edge,) * 3
    with pytest.raises(ValueError, match="2 \\* overlap < patch"):
        tiny_api.generate(blocks=(2, 2, 2), overlap=patch // 2, seed=0)


def test_each_z_tile_gets_the_height_and_profile_of_its_depth(height_api, monkeypatch):
    calls = []

    class Recorded(LRConditions):
        def __call__(self, index, tile):
            result = super().__call__(index, tile)
            calls.append((tile, result))
            return result

    monkeypatch.setattr(src.api, "LRConditions", Recorded)
    origin = 3.0
    blocks = {"blocks": (2, 1, 1), "overlap": 4}
    height_api.generate(seed=0, height_origin=origin, **blocks)
    plan = height_api.plan(**blocks)
    depth = plan.output_shape[0]
    config = height_api.config
    spacing = config.data.crop_size / config.patch_size
    extent = config.data.height_extent
    whole_height = height_field((depth, 1, 1), origin, spacing, extent)[0, 0, :, 0, 0]
    spec = config.data.depth_profile
    knots, values = validate_profile(spec, height_api.num_phases)
    whole_profile = integrate_profile(
        knots, values, spec["interpolation"], depth, origin, spacing, extent, None
    )[0]
    starts = sorted({tile.source[0].start for tile, _ in calls})
    assert len(starts) == 2
    second = next(item for item in calls if item[0].source[0].start == starts[1])
    tile, result = second
    # Output cells this tile owns, in output and in tile coordinates.
    first = max(tile.target[0].start, plan.margin)
    last = min(tile.target[0].stop, plan.margin + depth)
    output = slice(first - plan.margin, last - plan.margin)
    local = slice(first - tile.source[0].start, last - tile.source[0].start)
    assert output.stop > output.start
    torch.testing.assert_close(
        result["height"][0, 0, local, 0, 0], whole_height[output]
    )
    torch.testing.assert_close(
        result["profile"][0, :, local], whole_profile[:, output], atol=1e-5, rtol=0
    )
    torch.testing.assert_close(result["vf"], result["profile"].mean(-1))


def test_a_height_outside_the_source_is_rejected(height_api):
    extent = height_api.config.data.height_extent
    with pytest.raises(ValueError, match="outside the source image height"):
        height_api.generate(seed=0, height_origin=extent - 1)
    with pytest.raises(ValueError, match="outside the source image height"):
        height_api.generate(seed=0, height_origin=-1)


def test_an_output_smaller_than_the_patch_is_rejected(tiny_api):
    with pytest.raises(ValueError, match="at least the patch"):
        tiny_api.generate(shape=tiny_api.input_size - 1, seed=0)


def test_blocks_and_shape_together_are_rejected(tiny_api):
    with pytest.raises(ValueError, match="only one of blocks and shape"):
        tiny_api.generate(blocks=(2, 2, 2), shape=(32, 32, 32), seed=0)
