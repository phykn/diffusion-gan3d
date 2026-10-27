from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import src.train.run.low_res as low_res
from src.build.trainer import build_trainer
from src.config.files import load_yaml
from src.config.train import normalize_train_config
from src.plane import PLANES
from src.train.selection import CheckpointSelection, weights_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> dict:
    folder = tmp_path / "slices"
    folder.mkdir()
    for index in range(2):
        Image.fromarray(
            ((np.indices((8, 8)).sum(0) + index) % 3).astype(np.uint8)
        ).save(folder / f"{index}.png")
    cfg = load_yaml(PROJECT_ROOT / "tests/fixtures/config/train/low_res.yaml")
    cfg["data"] = {
        "domains": {0: {plane: [str(folder)] for plane in PLANES}},
        "crop_size": 8,
        "num_phases": 3,
        "lo_res_size": 8,
    }
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"]["channels"] = [4, 8]
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["train"].update(
        real_batch_size=2,
        slice_pairs_per_plane=2,
        mixed_precision=False,
        weights_every_steps=1,
        archive_every_steps=1,
    )
    cfg["selection"] = {"samples": 2}
    return cfg


def _selection(height_enabled=False):
    images = torch.nn.functional.one_hot(torch.randint(0, 2, (4, 4, 4)), 2)
    plane = {
        "images": images.movedim(-1, 1).float(),
        "height_origins": [0.0, 1.0, 2.0, 3.0],
        "height_extents": [8.0] * 4,
    }
    reference = {0: {axis: plane for axis in (0, 1, 2)}}
    return CheckpointSelection(reference, 4, height_enabled)


def _entry(step, structure):
    return {
        "step": step,
        "two_point_mae": 0.0,
        "chord_tv": 0.0,
        "cluster_error": 0.0,
        "structure": structure,
    }


def test_best_step_has_the_lowest_structure():
    selection = _selection()
    assert selection.best() is None
    assert not selection.export(Path("unused"))
    selection.history = [_entry(1, 10.0), _entry(2, 4.0), _entry(3, 6.0)]
    assert selection.best()["step"] == 2
    selection.history.append(_entry(4, 3.5))
    assert selection.best()["step"] == 4


def test_height_conditioned_volumes_take_side_crop_depths():
    assert _selection().heights(0) == [{}] * 4
    assert _selection(height_enabled=True).heights(0) == [
        {"height_origin": origin, "height_extent": 8.0}
        for origin in (0.0, 0.0, 1.0, 1.0)
    ]


def test_reference_and_scores_are_fixed_and_leave_training_rng_alone(tmp_path):
    torch.set_num_threads(1)
    cfg = _config(tmp_path)
    first = build_trainer(cfg, torch.device("cpu"))
    second = build_trainer(cfg, torch.device("cpu"))
    for axis, plane in first.selection.reference[0].items():
        assert torch.equal(
            plane["images"], second.selection.reference[0][axis]["images"]
        )
    torch.manual_seed(3)
    np.random.seed(3)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()[1]

    scores = first.selection.evaluate(first)

    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state)
    assert set(scores) == {"two_point_mae", "chord_tv", "cluster_error", "structure"}
    assert all(np.isfinite(value) and value >= 0 for value in scores.values())
    assert first.selection.evaluate(first) == scores


def test_runs_export_the_best_step_and_resume_keeps_its_weights(tmp_path, monkeypatch):
    scores = {
        1: _entry(1, 10.0),
        2: _entry(2, 4.0),
        3: _entry(3, 6.0),
        4: _entry(4, 3.5),
    }

    def evaluate(self, trainer):
        entry = dict(scores[trainer.completed_steps])
        entry.pop("step")
        return entry

    monkeypatch.setattr(CheckpointSelection, "evaluate", evaluate)
    monkeypatch.setattr(low_res, "PROJECT_ROOT", tmp_path)
    cfg = _config(tmp_path)
    first = low_res.run_low_res_train(
        "cpu", config=cfg, run_dir=tmp_path / "first", steps=2
    )
    best = weights_path(first, 2)
    # A new best replaces the weights of the one before it.
    assert not weights_path(first, 1).exists()
    assert (first / "generator.pt").read_bytes() == best.read_bytes()
    payload = torch.load(first / "checkpoints" / "step_00000002.pt", weights_only=True)
    assert [entry["step"] for entry in payload["selection"]["history"]] == [1, 2]

    resumed = low_res.run_low_res_train(
        "cpu", resume=first / "checkpoints", run_dir=tmp_path / "resumed", steps=3
    )
    # Step 3 was never best, so its weights were never kept.
    assert weights_path(resumed, 2).read_bytes() == best.read_bytes()
    assert not weights_path(resumed, 3).exists()
    assert (resumed / "generator.pt").read_bytes() == best.read_bytes()
    report = load_yaml(resumed / "selection" / "scores.yaml")
    assert report["best_step"] == 2
    assert [entry["step"] for entry in report["steps"]] == [1, 2, 3]

    final = low_res.run_low_res_train(
        "cpu", resume=resumed / "checkpoints", run_dir=tmp_path / "final", steps=4
    )
    assert not weights_path(final, 2).exists()
    assert (final / "generator.pt").read_bytes() == weights_path(final, 4).read_bytes()


def test_resume_requires_the_best_weights(tmp_path, monkeypatch):
    def evaluate(self, trainer):
        entry = _entry(0, 1.0)
        entry.pop("step")
        return entry

    monkeypatch.setattr(CheckpointSelection, "evaluate", evaluate)
    monkeypatch.setattr(low_res, "PROJECT_ROOT", tmp_path)
    first = low_res.run_low_res_train(
        "cpu", config=_config(tmp_path), run_dir=tmp_path / "first", steps=1
    )
    weights_path(first, 1).unlink()
    with pytest.raises(FileNotFoundError, match="selection weights"):
        low_res.run_low_res_train(
            "cpu", resume=first, run_dir=tmp_path / "resumed", steps=2
        )


def test_selection_without_scores_exports_the_latest_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(low_res, "PROJECT_ROOT", tmp_path)
    cfg = _config(tmp_path)
    cfg["selection"]["enabled"] = False
    run = low_res.run_low_res_train(
        "cpu", config=cfg, run_dir=tmp_path / "plain", steps=1
    )
    assert not (run / "selection").exists()
    exported = torch.load(run / "generator.pt", weights_only=True)
    payload = torch.load(run / "checkpoints" / "step_00000001.pt", weights_only=True)
    assert payload["selection"] is None
    for key, value in payload["ema"].items():
        assert torch.equal(exported[key], value)


def test_selection_settings_are_lr_only_and_validated():
    assert normalize_train_config({})["selection"] == {"enabled": True, "samples": 8}
    with pytest.raises(ValueError, match="selection"):
        normalize_train_config({"selection": {"samples": 2}}, "sr")
    for key, value in (
        ("enabled", "yes"),
        ("samples", 0),
        ("samples", 1.5),
    ):
        with pytest.raises(ValueError, match=f"selection.{key}"):
            normalize_train_config({"selection": {key: value}})
