import pytest
import torch

import src.api
import src.sampling.conditions
from src.api import LowResolutionAPI, SuperResolutionAPI
from src.data.height import build_heights
from src.data.profile import integrate_profile, validate_profile
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
        def __call__(self, tile):
            result = super().__call__(tile)
            calls.append((tile, result))
            return result

    monkeypatch.setattr(src.api, "LRConditions", Recorded)
    # A source taller than the training images keeps both z margins whole.
    origin, extent = 20.0, 200.0
    blocks = {"blocks": (2, 1, 1), "overlap": 4, "height_extent": extent}
    height_api.generate(seed=0, height_origin=origin, **blocks)
    plan = height_api.plan(height_origin=origin, **blocks)
    depth = plan.output_shape[0]
    config = height_api.config
    spacing = config.data.crop_size / config.patch_size
    whole_height = build_heights((depth, 1, 1), origin, spacing, extent, None)[
        0, 0, :, 0, 0
    ]
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
    top = plan.offsets[0]
    first = max(tile.target[0].start, top)
    last = min(tile.target[0].stop, top + depth)
    output = slice(first - top, last - top)
    local = slice(first - tile.source[0].start, last - tile.source[0].start)
    assert output.stop > output.start
    torch.testing.assert_close(
        result["height"][0, 0, local, 0, 0], whole_height[output]
    )
    torch.testing.assert_close(
        result["profile"][0, :, local], whole_profile[:, output], atol=1e-5, rtol=0
    )
    torch.testing.assert_close(result["vf"], result["profile"].mean(-1))


def test_full_depth_sees_only_heights_inside_the_source(
    tiny_height_configs, random_export, height_api, monkeypatch
):
    # Training crops lie inside the image, so their heights never reach -1 or 1;
    # generation must not feed the clamped heights of cells past a z face.
    seen = []

    def record(*args):
        heights = build_heights(*args)
        seen.append(heights)
        return heights

    monkeypatch.setattr(src.sampling.conditions, "build_heights", record)
    config = height_api.config
    extent = config.data.height_extent
    depth = round(extent * config.patch_size / config.data.crop_size)
    low = height_api.generate_probs(
        shape=(depth, 16, 16), seed=0, height_origin=0.0, overlap=4
    )
    sr_config = tiny_height_configs["sr"]
    sr = SuperResolutionAPI(random_export(sr_config, "height_sr", lr=config), "cpu")
    sr.refine(low, seed=0, height_origin=0.0)
    values = torch.cat([heights.flatten() for heights in seen])
    assert values.abs().max() < 1


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


def test_generate_labels_and_fractions(tiny_lr_run):
    api = LowResolutionAPI(tiny_lr_run, device="cpu")
    volume = api.generate(shape=16, seed=0)
    assert volume.dtype == torch.uint8
    assert set(volume.unique().tolist()) <= {0, 1}
    probs = api.generate_probs(shape=16, seed=0)
    assert probs.shape == (2, 16, 16, 16)
    torch.testing.assert_close(probs.sum(0), torch.ones(16, 16, 16))
    with pytest.raises(ValueError, match="storage"):
        api.generate(storage="disk")
    with pytest.raises(ValueError, match="original crop"):
        api.prepare_image(torch.zeros(16, 16, dtype=torch.long))
    assert api.prepare_image(torch.zeros(32, 32, dtype=torch.long)).shape == (2, 16, 16)
