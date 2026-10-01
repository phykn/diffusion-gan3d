from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.storage import torch_save
from src.train.checkpoint import complete_checkpoint, resolve_checkpoint
from src.train.run.loop import save_checkpoint, save_weights


def trainer():
    model = torch.nn.Linear(2, 2)
    optim = SimpleNamespace(state_dict=lambda: {})
    return SimpleNamespace(
        cfg={},
        completed_steps=1,
        data_fingerprint={},
        updates={},
        denoiser=model,
        ema_denoiser=model,
        critics=torch.nn.ModuleDict({"xy": model}),
        denoiser_optim=optim,
        critic_optims={"xy": optim},
        scaler=optim,
        use_multi_anchor_next=False,
        anchor_bank=SimpleNamespace(entries=[]),
    )


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_training_keeps_checkpoints_without_rename_or_temporary_files(
    tmp_path, monkeypatch, stage
):
    def forbidden(*args, **kwargs):
        pytest.fail("Training saves must not use temporary files or rename/link/delete")

    monkeypatch.setattr("src.storage.tempfile.NamedTemporaryFile", forbidden)
    monkeypatch.setattr(Path, "replace", forbidden)
    monkeypatch.setattr(Path, "rename", forbidden)
    monkeypatch.setattr(Path, "unlink", forbidden)
    monkeypatch.setattr("src.storage.os.link", forbidden)
    monkeypatch.setattr("src.train.run.loop.time_ns", lambda: 100)
    model = trainer()
    save_checkpoint(model, tmp_path, stage)
    first = resolve_checkpoint(tmp_path)
    original = first.read_bytes()
    model.completed_steps = 2
    save_checkpoint(model, tmp_path, stage)
    latest = resolve_checkpoint(tmp_path / "checkpoints")
    assert latest != first
    assert first.read_bytes() == original
    assert torch.load(latest, weights_only=True)["step"] == 2
    # A second save at the same boundary also preserves all previous checkpoints.
    save_checkpoint(model, tmp_path, stage)
    assert len(list((tmp_path / "checkpoints").glob("step_*.pt"))) == 3
    assert not (tmp_path / "generator.pt").exists()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_interrupted_locked_write_leaves_previous_checkpoint_usable(
    tmp_path, monkeypatch, stage
):
    model = trainer()
    save_checkpoint(model, tmp_path, stage)
    previous = resolve_checkpoint(tmp_path)
    original = previous.read_bytes()

    def interrupted(payload, file):
        file.write(b"partial checkpoint")
        raise OSError("write failed")

    def locked(*args, **kwargs):
        raise PermissionError("locked during cleanup")

    model.completed_steps = 2
    monkeypatch.setattr(torch, "save", interrupted)
    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(OSError, match="write failed"):
        save_checkpoint(model, tmp_path, stage)
    assert resolve_checkpoint(tmp_path) == previous
    assert previous.read_bytes() == original
    partial = next((tmp_path / "checkpoints").glob("step_00000002_*.pt"))
    with pytest.raises(ValueError, match="incomplete"):
        resolve_checkpoint(partial)


def test_checkpoint_is_available_even_if_generator_export_fails(tmp_path, monkeypatch):
    model = trainer()
    save_checkpoint(model, tmp_path, "low_res")
    model.completed_steps = 2
    original = torch.save

    def fail_export(payload, file):
        if Path(file.name).name == "generator.pt":
            file.write(b"interrupted export")
            raise OSError("export failed")
        original(payload, file)

    monkeypatch.setattr(torch, "save", fail_export)
    save_checkpoint(model, tmp_path, "low_res")
    with pytest.raises(OSError, match="export failed"):
        save_weights(model, tmp_path, "low_res")
    assert torch.load(resolve_checkpoint(tmp_path), weights_only=True)["step"] == 2


@pytest.mark.parametrize("damage", ["absent", "partial_marker", "truncated"])
def test_directory_resume_skips_unfinished_checkpoint(tmp_path, damage):
    root = tmp_path / "checkpoints"
    previous = torch_save({"step": 9}, root / "step_00000009_100.pt", overwrite=False)
    complete_checkpoint(previous)
    latest = torch_save({"step": 10}, root / "step_00000010_200.pt", overwrite=False)
    if damage == "partial_marker":
        latest.with_suffix(".complete").write_text("12", encoding="ascii")
    elif damage == "truncated":
        complete_checkpoint(latest)
        latest.write_bytes(b"truncated")
    assert resolve_checkpoint(tmp_path) == previous.resolve()
    with pytest.raises(ValueError, match="incomplete"):
        resolve_checkpoint(latest)


def test_latest_checkpoint_orders_steps_numerically(tmp_path):
    root = tmp_path / "checkpoints"
    for step, timestamp in ((9, 900), (10, 100), (10, 200)):
        path = torch_save({}, root / f"step_{step}_{timestamp}.pt", overwrite=False)
        complete_checkpoint(path)
    assert resolve_checkpoint(tmp_path).name == "step_10_200.pt"


def test_unmarked_checkpoint_is_rejected_as_a_file_and_directory_fallback(tmp_path):
    path = torch_save({"step": 1}, tmp_path / "checkpoints" / "last.pt")
    with pytest.raises(FileNotFoundError, match="No completed"):
        resolve_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        resolve_checkpoint(path)


def test_renamed_completed_checkpoint_can_be_selected_explicitly(tmp_path):
    path = torch_save({"step": 1}, tmp_path / "training.pt")
    complete_checkpoint(path)
    assert resolve_checkpoint(path) == path.resolve()


def test_exclusive_write_cannot_damage_existing_checkpoint(tmp_path):
    path = torch_save({"step": 1}, tmp_path / "checkpoint.pt", overwrite=False)
    with pytest.raises(FileExistsError):
        torch_save({"step": 2}, path, overwrite=False)
    assert torch.load(path, weights_only=True)["step"] == 1


def test_directory_without_completed_checkpoint_reports_missing(tmp_path):
    torch_save({}, tmp_path / "step_00000001_100.pt", overwrite=False)
    with pytest.raises(FileNotFoundError, match="No completed"):
        resolve_checkpoint(tmp_path)
