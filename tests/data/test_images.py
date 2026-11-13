"""Label images, phase count, height and depth profile."""

import numpy as np
import pytest
from PIL import Image

from src.data.images import infer_num_phases, load_images


def _save(path, values, **options):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values)).save(path, **options)
    return path


def test_missing_label_names_the_file(tmp_path):
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 2
    path = _save(tmp_path / "gap.png", image)
    with pytest.raises(ValueError) as error:
        infer_num_phases(load_images({"xy": [tmp_path]}))
    assert "label 1 never appears below the largest label 2" in str(error.value)
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
