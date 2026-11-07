"""Label images, phase count, height and depth profile (SPEC Label images)."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.config.io import PROJECT_ROOT
from src.config.sample import load_sample
from src.data.crops import resize_phases
from src.data.height import depth_profile, height_extent
from src.data.images import infer_num_phases, load_images

PARITY = Path(__file__).parents[1] / "fixtures" / "parity"
REFERENCE = yaml.safe_load((PARITY / "images.yaml").read_text())


def _save(path: Path, values, **options) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values)).save(path, **options)
    return path


def _sample_planes(sample: str):
    if (
        sample == "simul_anode"
        and not (PROJECT_ROOT / "data/generated/simul_anode/slices/0").is_dir()
    ):
        pytest.skip("data/generated/simul_anode is not generated here.")
    folders = {
        plane: [PROJECT_ROOT / folder for folder in folders]
        for plane, folders in load_sample(sample)["images"].items()
    }
    return load_images(folders)


@pytest.mark.parametrize("sample", ["default", "simul_anode"])
def test_images_match_v1(sample):
    planes = _sample_planes(sample)
    expected = REFERENCE[sample]
    num_phases = infer_num_phases(planes)
    assert num_phases == expected["num_phases"]
    assert height_extent(planes) == expected["height_extent"]
    assert depth_profile(planes, num_phases, 16) == expected["depth_profile"]


def test_depth_profile_bins():
    planes = _sample_planes("default")
    profile = depth_profile(planes, 2, 8)
    assert len(profile["points"]) == 9
    assert profile["points"][-1][1] == profile["points"][-2][1]
    assert all(abs(sum(row) - 1) < 1e-9 for _, row in profile["points"])


def test_missing_label_names_the_file(tmp_path):
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 2
    path = _save(tmp_path / "gap.png", image)
    with pytest.raises(ValueError) as error:
        infer_num_phases(load_images({"xy": [tmp_path]}))
    # The v1 message, then the file holding the largest label.
    assert REFERENCE["missing_label_error"] in str(error.value)
    assert str(path) in str(error.value)
    assert load_images({"xy": [tmp_path]}, num_phases=3)["xy"].all_images()


def test_bad_images_name_the_file(tmp_path):
    frames = [Image.fromarray(np.zeros((4, 4), dtype=np.uint8)) for _ in range(2)]
    stack = tmp_path / "stack" / "frames.tif"
    stack.parent.mkdir()
    frames[0].save(stack, save_all=True, append_images=frames[1:])
    wide = _save(tmp_path / "wide" / "wide.tif", np.zeros((4, 4), dtype=np.uint16))
    for path in (stack, wide):
        with pytest.raises(ValueError, match=path.name):
            load_images({"xy": [path.parent]})


def test_only_direct_images_are_read(tmp_path):
    _save(tmp_path / "a.png", np.zeros((4, 4), dtype=np.uint8))
    _save(tmp_path / "nested" / "b.png", np.ones((4, 4), dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    images = load_images({"xy": [tmp_path]})["xy"]
    assert [path.name for path in images.paths[0]] == ["a.png"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no images"):
        load_images({"xy": [empty]})


def test_resize_matches_v1():
    cases = torch.load(PARITY / "resize.pt", weights_only=True)
    for name, case in cases.items():
        result = resize_phases(case["input"], tuple(case["shape"]))
        torch.testing.assert_close(result, case["output"], atol=1e-6, rtol=0, msg=name)
