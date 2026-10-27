from pathlib import Path

import pytest
import torch

import src.storage as storage
from src.model.critic import PairCritic2D
from src.storage import (
    load_probabilities,
    load_volume,
    save_model,
    save_probabilities,
    save_volume,
)
from src.train.run.bank import publish_bank
from src.train.state import export_generator


@pytest.mark.parametrize("damage", ["extra", "missing"])
def test_model_loading_rejects_incompatible_weights(tmp_path, damage):
    model = PairCritic2D(2, (4, 8), 8, 1)
    state = model.state_dict()
    if damage == "extra":
        state["unused.weight"] = torch.ones(1)
        message = "Unexpected key"
    else:
        del state["output.weight"]
        message = "Missing key"
    path = tmp_path / "critic.pt"
    torch.save(state, path)
    restored = PairCritic2D(2, (4, 8), 8, 1)
    with pytest.raises(RuntimeError, match=message):
        storage.load_model(path, restored)


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


@pytest.mark.parametrize("kind", ["model", "sr"])
def test_exports_overwrite_directly(tmp_path, kind):
    path = tmp_path / "generator.pt"
    path.write_bytes(b"previous export")
    model = torch.nn.Linear(2, 2)
    if kind == "model":
        save_model(path, model)
        expected = model.state_dict()
    else:
        export_generator({"stage": "sr"}, 3, model.state_dict(), path)
        expected = {
            "format": "diffusion-gan3d.sr",
            "config": {"stage": "sr"},
            "step": 3,
            "model": model.state_dict(),
        }
    actual = torch.load(path, weights_only=True)
    if kind == "sr":
        assert actual.pop("format") == expected.pop("format")
        assert actual.pop("config") == expected.pop("config")
    torch.testing.assert_close(actual, expected)
    assert list(tmp_path.iterdir()) == [path]


def test_failed_bank_write_can_be_retried_without_publishing_partial_file(
    tmp_path, monkeypatch
):
    def fail(payload, file):
        file.write(b"partial snapshot")
        raise OSError("disk full")

    cfg = {"data": {}, "source": {}}
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="disk full"):
            publish_bank({"step": 7}, cfg, tmp_path, 7)
    assert list((tmp_path / "lr_bank").iterdir()) == []
    publish_bank({"step": 7}, cfg, tmp_path, 7)
    saved = torch.load(cfg["source"]["bank"], weights_only=True)
    assert saved["step"] == 7
