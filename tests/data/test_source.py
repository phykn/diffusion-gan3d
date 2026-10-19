import numpy as np
import pytest
from PIL import Image

from src.data.source import (
    collect_image_groups,
    infer_depth_profiles,
    infer_height_extents,
)


def test_height_inference_ignores_directories_with_image_extensions(tmp_path):
    Image.fromarray(np.zeros((12, 10), dtype=np.uint8)).save(tmp_path / "image.png")
    (tmp_path / "folder.png").mkdir()
    data = {"domains": {0: {"xz": [str(tmp_path)]}}}

    assert infer_height_extents(data) == {0: 12}


def test_image_groups_preserve_folder_boundaries_and_exclude_heldout_files(tmp_path):
    folders = [tmp_path / name for name in ("first", "second", "heldout")]
    for folder in folders:
        folder.mkdir()
        for name in ("b.png", "a.PNG"):
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(folder / name)
    heldout = [folders[0] / "b.png", *folders[2].iterdir()]
    data = {
        "domains": {0: {"xy": folders}},
        "split": {"validation_files": [str(path.resolve()) for path in heldout]},
    }

    groups = collect_image_groups(data)

    assert groups == {
        0: {0: ((folders[0] / "a.PNG",), (folders[1] / "a.PNG", folders[1] / "b.png"))}
    }


def test_split_cannot_reference_an_image_outside_the_configured_sources(tmp_path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "image.png")
    data = {
        "domains": {0: {"xy": [tmp_path]}},
        "split": {"validation_regions": {str(tmp_path / "missing.png"): [0, 0, 2, 2]}},
    }

    with pytest.raises(ValueError, match="split paths must identify images"):
        collect_image_groups(data)


def test_height_inference_uses_training_images_and_checks_saved_extents(tmp_path):
    for name, height in (("training.png", 12), ("validation.png", 20)):
        Image.fromarray(np.zeros((height, 10), dtype=np.uint8)).save(tmp_path / name)
    data = {
        "domains": {0: {"xz": [tmp_path]}},
        "split": {"validation_files": [str((tmp_path / "validation.png").resolve())]},
    }

    assert infer_height_extents(data) == {0: 12}
    assert "height_extents" not in data
    data["height_extents"] = {0: 20}
    with pytest.raises(ValueError, match="differs from saved height_extents"):
        infer_height_extents(data)


def test_height_inference_uses_side_rows_and_ignores_xy_size(tmp_path):
    planes = {}
    for plane, shape in (("xy", (7, 19)), ("xz", (12, 10)), ("yz", (12, 17))):
        folder = tmp_path / plane
        folder.mkdir()
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(folder / "image.png")
        planes[plane] = [folder]
    assert infer_height_extents({"domains": {0: planes}}) == {0: 12}


def test_height_inference_requires_side_images(tmp_path):
    Image.fromarray(np.zeros((12, 10), dtype=np.uint8)).save(tmp_path / "image.png")
    with pytest.raises(ValueError, match="full-height side images"):
        infer_height_extents({"domains": {0: {"xy": [tmp_path]}}})


def test_depth_profiles_average_side_images_in_z_bands(tmp_path):
    image = np.zeros((16, 6), dtype=np.uint8)
    image[8:] = 1
    image[12:, :3] = 0
    for folder in ("xz", "yz"):
        (tmp_path / folder).mkdir()
        Image.fromarray(image).save(tmp_path / folder / "a.png")
    data = {
        "num_phases": 2,
        "domains": {0: {"xz": [str(tmp_path / "xz")], "yz": [str(tmp_path / "yz")]}},
    }

    profile = infer_depth_profiles(data, bins=2)[0]

    assert profile["axis"] == "z" and profile["interpolation"] == "constant"
    assert profile["points"] == [
        [0.0, [1.0, 0.0]],
        [0.5, [0.25, 0.75]],
        [1.0, [0.25, 0.75]],
    ]


def test_depth_profiles_skip_domains_with_mixed_heights(tmp_path):
    Image.fromarray(np.zeros((16, 6), dtype=np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((12, 6), dtype=np.uint8)).save(tmp_path / "b.png")
    data = {"num_phases": 2, "domains": {0: {"xz": [str(tmp_path)]}}}

    assert infer_depth_profiles(data, bins=2) == {0: None}
