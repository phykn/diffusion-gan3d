from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import src.train.run.low_res as low_res
import src.train.run.sr as sr_run
import src.train.selection as selection_module
from src.build.trainer import build_trainer
from src.config.files import load_yaml, save_yaml
from src.config.train import normalize_train_config
from src.evaluate.structure import SectionErrors, section_errors
from src.plane import PLANES
from src.predict.sr.inference import SuperResolutionAPI, load_sr_export
from src.train.run.loop import run_train
from src.train.selection import (
    CheckpointSelection,
    SRCheckpointSelection,
    inputs_path,
    pick_sections,
    weights_path,
)

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


def test_selection_settings_apply_to_both_stages_and_are_validated():
    for stage in ("low_res", "sr"):
        assert normalize_train_config({}, stage)["selection"] == {
            "enabled": True,
            "samples": 8,
        }
        for key, value in (
            ("enabled", "yes"),
            ("samples", 0),
            ("samples", 1.5),
        ):
            with pytest.raises(ValueError, match=f"selection.{key}"):
                normalize_train_config({"selection": {key: value}}, stage)


def test_sections_are_a_fixed_count_spread_over_every_volume(monkeypatch):
    monkeypatch.setattr(selection_module, "SECTIONS", 6)
    volumes = [
        torch.full((4, 5, 5), index, dtype=torch.uint8).index_put_(
            (torch.arange(4),),
            torch.arange(4, dtype=torch.uint8)[:, None, None] + 10 * index,
        )
        for index in range(3)
    ]
    sections = pick_sections(volumes, 0, 40).argmax(1)[:, 0, 0]
    # Twelve slices in all; six evenly spaced ones reach the first and last.
    assert sections.tolist() == [0, 2, 10, 13, 21, 23]
    assert pick_sections(volumes[:1], 0, 40).shape == (4, 40, 5, 5)
    # Every axis keeps the in-plane order of a full movedim.
    volume = torch.randint(0, 40, (4, 5, 6), dtype=torch.uint8)
    for axis in (1, 2):
        full = volume.long().movedim(axis, 0)
        assert torch.equal(pick_sections([volume], axis, 40).argmax(1), full)


def test_cached_real_statistics_match_direct_section_errors():
    selection = _selection()
    volumes = [torch.randint(0, 2, (4, 4, 4), dtype=torch.uint8) for _ in range(2)]
    scores = selection.score(lambda domain: volumes)
    plane = selection.reference[0][0]["images"]
    fake = torch.nn.functional.one_hot(torch.stack(volumes).long(), 2)
    direct = [
        section_errors(
            plane, fake.movedim(axis + 1, 1).flatten(0, 1).movedim(-1, 1).float()
        )
        for axis in (0, 1, 2)
    ]
    for index, name in enumerate(SectionErrors._fields):
        assert scores[name] == pytest.approx(np.mean([e[index] for e in direct]))


def test_sr_inputs_keep_their_bank_heights_and_cycle_when_fewer():
    reference = _selection().reference
    bank = {0: torch.rand(2, 2, 4, 4, 4)}
    origins = {0: torch.tensor([1.0, 2.0])}
    selection = SRCheckpointSelection.from_bank(
        reference, 3, bank, origins, {0: torch.tensor([9.0, 8.0])}
    )
    assert torch.equal(selection.inputs["volumes"][0], bank[0])
    assert selection.heights(0) == [
        {"height_origin": 1.0, "height_extent": 9.0},
        {"height_origin": 2.0, "height_extent": 8.0},
        {"height_origin": 1.0, "height_extent": 9.0},
    ]
    plain = SRCheckpointSelection.from_bank(reference, 3, bank, None, {0: 1.0})
    assert plain.heights(0) == [{}] * 3


def _train_base(tmp_path: Path) -> Path:
    cfg = _config(tmp_path)
    cfg["selection"]["enabled"] = False
    base = tmp_path / "base"
    base.mkdir()
    save_yaml(base / "train.yaml", cfg)
    run_train(build_trainer(cfg, torch.device("cpu")), 1, 1, base)
    return base


def _sr_config(tmp_path: Path, base: Path) -> dict:
    cfg = load_yaml(PROJECT_ROOT / "tests/fixtures/config/train/sr.yaml")
    cfg["data"] = {**_config(tmp_path / "unused")["data"], "hi_res_size": 16}
    cfg["data"]["domains"] = load_yaml(base / "train.yaml")["data"]["domains"]
    cfg["data"]["crop_size"] = 8
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
    cfg["lr_bank"].update(samples_per_domain=2, refresh_every_steps=1)
    cfg["selection"] = {"samples": 3}
    cfg["source"] = {"weights": str(base)}
    return cfg


def test_sr_runs_score_fixed_lr_inputs_and_export_the_best_step(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    (tmp_path / "unused").mkdir()
    base = _train_base(tmp_path)
    monkeypatch.setattr(sr_run, "PROJECT_ROOT", tmp_path)
    structure = {1: 10.0, 2: 4.0, 3: 6.0}
    scored = []
    original = SRCheckpointSelection.evaluate

    def evaluate(self, trainer):
        entry = original(self, trainer)
        scored.append(self.inputs["volumes"][0].clone())
        return {**entry, "structure": structure[trainer.completed_steps]}

    monkeypatch.setattr(SRCheckpointSelection, "evaluate", evaluate)
    first = sr_run.run_sr_train(
        "cpu", config=_sr_config(tmp_path, base), run_dir=tmp_path / "first", steps=2
    )
    initial = torch.load(first / "lr_bank" / "step_00000000.pt", weights_only=True)
    inputs = torch.load(inputs_path(first), weights_only=True)
    assert torch.equal(inputs["volumes"][0], initial["volumes"][0][:2])
    report = load_yaml(first / "selection" / "scores.yaml")
    assert report["best_step"] == 2
    assert all(0 <= entry["lr_mse"] <= 1 for entry in report["steps"])
    assert not weights_path(first, 1).exists()
    exported = load_sr_export(first / "generator.pt")
    assert exported["step"] == 2
    assert (first / "generator.pt").read_bytes() == weights_path(first, 2).read_bytes()

    resumed = sr_run.run_sr_train(
        "cpu", resume=first / "checkpoints", run_dir=tmp_path / "resumed", steps=3
    )
    # Bank refreshes and the resume never change what is scored.
    assert len(scored) == 3
    assert all(torch.equal(volumes, inputs["volumes"][0]) for volumes in scored)
    assert inputs_path(resumed).read_bytes() == inputs_path(first).read_bytes()
    assert load_sr_export(resumed / "generator.pt")["step"] == 2
    assert SuperResolutionAPI(resumed, "cpu").refine(
        inputs["volumes"][0][0], seed=0
    ).shape == (16, 16, 16)


def test_sr_resume_requires_the_scored_inputs(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    (tmp_path / "unused").mkdir()
    base = _train_base(tmp_path)
    monkeypatch.setattr(sr_run, "PROJECT_ROOT", tmp_path)
    first = sr_run.run_sr_train(
        "cpu", config=_sr_config(tmp_path, base), run_dir=tmp_path / "first", steps=1
    )
    inputs_path(first).unlink()
    with pytest.raises(FileNotFoundError, match="selection inputs"):
        sr_run.run_sr_train("cpu", resume=first, run_dir=tmp_path / "resumed", steps=2)
