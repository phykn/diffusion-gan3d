import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.storage as storage
from src.model.critic import PairCritic2D
from src.storage import (
    atomic_torch_save,
    load_probabilities,
    load_volume,
    save_model,
    save_probabilities,
    save_volume,
)
from src.train.run.bank import save_bank
from src.train.state import export_sr


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


@pytest.mark.parametrize("existing", [False, True])
def test_failed_probability_save_preserves_previous_file_and_cleans_up(
    tmp_path, monkeypatch, existing
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
        save_probabilities(torch.ones(2, 1, 1, 1) / 2, path)
    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    if existing:
        assert path.read_bytes() == previous


@pytest.mark.parametrize("kind", ["model", "sr"])
def test_exports_overwrite_directly_without_temporary_files(
    tmp_path, monkeypatch, kind
):
    path = tmp_path / "generator.pt"
    path.write_bytes(b"previous export")

    def no_temporary(*args, **kwargs):
        pytest.fail("Direct exports must not create temporary files")

    monkeypatch.setattr(storage.tempfile, "NamedTemporaryFile", no_temporary)
    model = torch.nn.Linear(2, 2)
    if kind == "model":
        save_model(path, model)
        expected = model.state_dict()
    else:
        export_sr(SimpleNamespace(cfg={}, completed_steps=3, ema_denoiser=model), path)
        expected = {
            "format": "diffusion-gan3d.sr",
            "config": {},
            "step": 3,
            "model": model.state_dict(),
        }
    actual = torch.load(path, weights_only=True)
    if kind == "sr":
        assert actual.pop("format") == expected.pop("format")
    torch.testing.assert_close(actual, expected)
    assert list(tmp_path.iterdir()) == [path]


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


def locked_file_error(code=32):
    error = PermissionError("file locked")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_atomic_save_retries_publication_without_serializing_again(
    tmp_path, monkeypatch, code
):
    path = tmp_path / "last.pt"
    atomic_torch_save({"step": 1}, path)
    save = torch.save
    publish = Path.replace
    attempts = []
    writers = []
    delays = []

    def capture_save(payload, file):
        writers.append(file)
        save(payload, file)

    def busy_publish(source, destination):
        assert writers[0].closed
        assert torch.load(source, weights_only=True) == {"step": 2}
        attempts.append(source)
        if len(attempts) <= 3:
            assert torch.load(path, weights_only=True) == {"step": 1}
            raise locked_file_error(code)
        return publish(source, destination)

    monkeypatch.setattr(torch, "save", capture_save)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    monkeypatch.setattr(Path, "replace", busy_publish)
    assert atomic_torch_save({"step": 2}, path) == path
    assert len(writers) == 1
    assert len(set(attempts)) == 1
    assert delays == [0.1, 0.2, 0.4]
    assert torch.load(path, weights_only=True) == {"step": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_lock_preserves_complete_file(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_torch_save({"step": 1}, path)
    error = locked_file_error()
    delays = []

    def busy_publish(source, destination):
        raise error

    monkeypatch.setattr(storage.time, "sleep", delays.append)
    monkeypatch.setattr(Path, "replace", busy_publish)
    with pytest.raises(PermissionError) as caught:
        atomic_torch_save({"step": 2}, path)
    assert caught.value is error
    assert len(delays) == 10
    assert sum(delays) == pytest.approx(7.5)
    assert torch.load(path, weights_only=True) == {"step": 1}
    (recovery,) = tmp_path.glob(".last.pt.*.tmp")
    assert torch.load(recovery, weights_only=True) == {"step": 2}
    assert str(recovery.resolve()) in error.__notes__[0]


@pytest.mark.parametrize("save_fails", [False, True])
def test_cleanup_lock_does_not_fail_published_save_or_mask_write_error(
    tmp_path, monkeypatch, caplog, save_fails
):
    path = tmp_path / "snapshot.pt"
    delays = []
    write_error = OSError("disk full")

    def busy_unlink(file, **kwargs):
        raise locked_file_error()

    def failed_save(payload, file):
        file.write(b"partial")
        raise write_error

    with monkeypatch.context() as patch:
        patch.setattr(storage.time, "sleep", delays.append)
        patch.setattr(Path, "unlink", busy_unlink)
        if save_fails:
            patch.setattr(torch, "save", failed_save)
            with pytest.raises(OSError) as caught:
                atomic_torch_save({"step": 2}, path)
            assert caught.value is write_error
            assert not path.exists()
        else:
            assert atomic_torch_save({"step": 2}, path) == path
            assert torch.load(path, weights_only=True) == {"step": 2}
    assert len(delays) == 10
    assert "Could not remove temporary file" in caplog.text


def test_non_lock_failure_is_not_retried(tmp_path, monkeypatch):
    delays = []

    def denied(source, destination):
        raise locked_file_error(112)

    monkeypatch.setattr(storage.time, "sleep", delays.append)
    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError):
        atomic_torch_save({}, tmp_path / "last.pt")
    assert delays == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
@pytest.mark.parametrize("locked_target", ["source", "destination"])
def test_atomic_save_recovers_from_real_windows_lock(
    tmp_path, monkeypatch, locked_target
):
    path = tmp_path / "last.pt"
    atomic_torch_save({"step": 1}, path)
    replace = Path.replace
    handles = []
    errors = []

    def replace_with_lock(source, destination):
        if not handles:
            locked_path = source if locked_target == "source" else destination
            handles.append(locked_path.open("rb"))
        try:
            return replace(source, destination)
        except OSError as error:
            errors.append(error.winerror)
            raise

    def release_lock(delay):
        assert torch.load(path, weights_only=True) == {"step": 1}
        handles[0].close()

    monkeypatch.setattr(Path, "replace", replace_with_lock)
    monkeypatch.setattr(storage.time, "sleep", release_lock)
    try:
        assert atomic_torch_save({"step": 2}, path) == path
    finally:
        for handle in handles:
            handle.close()
    assert errors == [32 if locked_target == "source" else 5]
    assert torch.load(path, weights_only=True) == {"step": 2}
    assert list(tmp_path.iterdir()) == [path]
