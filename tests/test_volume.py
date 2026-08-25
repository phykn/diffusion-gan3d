from pathlib import Path

import torch

from src.utils import load_volume, save_volume


def test_label_volume_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.tiff"
    expected = (torch.arange(24).reshape(2, 3, 4) % 3).to(torch.long)

    save_volume(expected, path)
    actual = load_volume(path)

    assert torch.equal(actual, expected)
    assert actual.dtype == torch.long
