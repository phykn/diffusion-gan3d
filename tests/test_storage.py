from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.storage import (
    atomic_torch_save,
    load_probabilities,
    load_volume,
    save_model,
    save_probabilities,
    save_volume,
)
from src.train.run.bank import save_bank
from src.train.sr import export_sr


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


@pytest.mark.parametrize("kind", ["model", "probabilities", "sr"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_artifact_save_preserves_previous_file_and_cleans_up(
    tmp_path, monkeypatch, kind, existing
):
    path = tmp_path / "artifact.pt"
    previous = b"previous complete artifact"
    if existing:
        path.write_bytes(previous)

    def interrupted_save(payload, file):
        file.write(b"incomplete new artifact")
        raise OSError("disk write failed")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="disk write failed"):
        if kind == "model":
            save_model(path, torch.nn.Linear(2, 2))
        elif kind == "probabilities":
            save_probabilities(torch.ones(2, 1, 1, 1) / 2, path)
        else:
            trainer = SimpleNamespace(
                cfg={}, completed_steps=3, ema_denoiser=torch.nn.Linear(2, 2)
            )
            export_sr(trainer, path)
    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    if existing:
        assert path.read_bytes() == previous


def test_atomic_save_cleans_up_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    atomic_torch_save({"step": 1}, path)

    def failed_replace(source, destination):
        raise PermissionError("destination locked")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(PermissionError, match="destination locked"):
        atomic_torch_save({"step": 2}, path)
    assert torch.load(path, weights_only=True) == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_save_replaces_existing_artifact(tmp_path):
    path = tmp_path / "nested" / "model.pt"
    atomic_torch_save({"step": 1}, path)
    assert atomic_torch_save({"step": 2}, path) == path
    assert torch.load(path, weights_only=True) == {"step": 2}
    assert list(path.parent.iterdir()) == [path]


def test_failed_bank_write_can_be_retried_without_publishing_partial_file(
    tmp_path, monkeypatch
):
    def fail(payload, file):
        file.write(b"partial snapshot")
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="disk full"):
            save_bank(tmp_path, 7, {"step": 7})
    assert list((tmp_path / "lr_bank").iterdir()) == []
    saved = save_bank(tmp_path, 7, {"step": 7})
    assert torch.load(saved["bank"], weights_only=True) == {"step": 7}


def test_exclusive_save_preserves_concurrently_published_file(tmp_path, monkeypatch):
    import os

    path = tmp_path / "snapshot.pt"
    link = os.link

    def competing_link(source, destination):
        Path(destination).write_bytes(b"another complete snapshot")
        link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        atomic_torch_save({"step": 7}, path, overwrite=False)
    assert path.read_bytes() == b"another complete snapshot"
    assert list(tmp_path.iterdir()) == [path]
