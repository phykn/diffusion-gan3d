from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import RealDataset
from src.data.split import resolve_split, training_origin


def test_every_training_crop_avoids_validation_and_no_valid_origin_is_lost():
    shape, crop, held = (10, 12), 3, [4, 5, 2, 3]
    expected = {
        (y, x)
        for y in range(8)
        for x in range(10)
        if y + 3 <= 4 or y >= 6 or x + 3 <= 5 or x >= 8
    }
    actual = set()
    for index in range(len(expected)):
        with patch("src.data.split.np.random.randint", return_value=index):
            actual.add(training_origin(shape, crop, held))
    assert actual == expected


def test_heldout_crop_keeps_full_original_height(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((20, 16), dtype=np.uint8)).save(path)
    dataset = RealDataset(
        [[path]],
        4,
        2,
        2,
        plane="xz",
        validation_regions={str(path.resolve()): [0, 0, 10, 16]},
    )
    for _ in range(10):
        sample = dataset[path]
        assert sample["height_origin"] >= 10
        assert sample["height_extent"] == 20
        assert torch.equal(sample["source_shape"], torch.tensor([20, 16]))


def test_invalid_or_unusable_split_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_split({"validation_regions": {"a.png": [0, 0, -1, 2]}}, tmp_path)
    with pytest.raises(ValueError, match="no training crop"):
        training_origin((8, 8), 8, [0, 0, 2, 2])


def test_region_path_aliases_cannot_silently_replace_a_holdout(tmp_path):
    settings = {
        "validation_regions": {
            "image.png": [0, 0, 2, 2],
            "./image.png": [4, 4, 2, 2],
        }
    }
    with pytest.raises(ValueError, match="duplicate"):
        resolve_split(settings, tmp_path)
