from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from src.volume import load_volume, save_volume


def test_label_volume_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.tiff"
    expected = (torch.arange(24).reshape(2, 3, 4) % 3).to(torch.long)

    save_volume(expected, path)
    actual = load_volume(path, shape=(2, 3, 4), num_phases=3)

    assert torch.equal(actual, expected)
    assert actual.dtype == torch.long


def test_label_volume_rejects_implicit_resize(tmp_path: Path) -> None:
    path = tmp_path / "labels.tiff"
    tifffile.imwrite(path, np.zeros((5, 6, 7), dtype=np.uint8))

    with pytest.raises(ValueError, match="must have shape.*got"):
        load_volume(path, shape=(5, 5, 5), num_phases=2)


def test_label_volume_rejects_out_of_range_phase(tmp_path: Path) -> None:
    path = tmp_path / "labels.tiff"
    tifffile.imwrite(path, np.full((2, 2, 2), 2, dtype=np.uint8))

    with pytest.raises(ValueError, match="phases from 0 to 1"):
        load_volume(path, num_phases=2)
