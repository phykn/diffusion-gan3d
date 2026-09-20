from pathlib import Path

import pytest
import torch

from src.storage import load_probabilities, load_volume, save_probabilities, save_volume


def test_label_volume_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.tiff"
    expected = (torch.arange(24).reshape(2, 3, 4) % 3).to(torch.long)

    save_volume(expected, path)
    actual = load_volume(path)

    assert torch.equal(actual, expected)
    assert actual.dtype == torch.long


@pytest.mark.parametrize(
    "volume",
    [
        torch.tensor([[[-1]]]),
        torch.tensor([[[256]]]),
        torch.tensor([[[0.5]]]),
        torch.zeros(2, 2),
        torch.empty(0, 2, 2, dtype=torch.long),
    ],
)
def test_save_rejects_invalid_labels_before_writing(tmp_path, volume):
    path = tmp_path / "invalid.tiff"
    with pytest.raises(ValueError):
        save_volume(volume, path)
    assert not path.exists()


def test_fractional_volume_round_trip_preserves_channels(tmp_path):
    probs = torch.tensor([0.25, 0.75]).reshape(2, 1, 1, 1)
    path = tmp_path / "probs.pt"
    assert save_probabilities(probs, path) == path
    assert torch.equal(load_probabilities(path), probs)


@pytest.mark.parametrize(
    "probs", [torch.zeros(2, 2, 2), torch.zeros(2, 1, 1, 1, dtype=torch.long)]
)
def test_probability_save_rejects_label_tensors(tmp_path, probs):
    path = tmp_path / "probs.pt"
    with pytest.raises(ValueError):
        save_probabilities(probs, path)
    assert not path.exists()
