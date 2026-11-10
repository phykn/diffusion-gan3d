"""SR stage: exports, resume without the LR run, SR source checks."""

import shutil
from pathlib import Path

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train
from src.api import SuperResolutionAPI
from src.config.schema import update
from src.export import read_export
from src.train.run import resume_stage, train_stage
from src.train.sr import corrupt_coarse

TINY = Path(__file__).parents[1] / "fixtures" / "tiny"
LOSSES = ("loss/generator", "loss/generator_total", "loss/critic", "consistency")


def _losses(run_dir: Path) -> dict:
    events = EventAccumulator(str(run_dir / "tensorboard"))
    events.Reload()
    return {
        tag: {event.step: event.value for event in events.Scalars(tag)}
        for tag in LOSSES
    }


def test_a_copied_sr_export_refines_alone(tiny_sr_run, tmp_path):
    alone = tmp_path / "generator.pt"
    shutil.copy(tiny_sr_run / "generator.pt", alone)
    api = SuperResolutionAPI(alone, "cpu")
    low = api.lr.generate(size=16, seed=0)
    assert api.refine(low, seed=0).shape == (32, 32, 32)
    tile, _ = api.default_tiling()
    assert api.refine(low, seed=0, tile_size=tile).shape == (32, 32, 32)


def test_sr_resume_matches_an_unbroken_run_without_the_lr_run(
    tiny_configs, tiny_lr_run, tmp_path
):
    # The bank refreshes every step, so the bank must resume exactly too.
    lr = Path(shutil.copytree(tiny_lr_run, tmp_path / "copy-lr"))
    config = update(tiny_configs["sr"], "train.structure_every_steps", 0)
    whole = train_stage(
        update(config, "train.total_steps", 2), "cpu", tmp_path / "whole", lr
    )
    first = train_stage(config, "cpu", tmp_path / "first", lr)
    shutil.rmtree(lr)
    stage, second = resume_stage(first, "cpu", tmp_path / "second", steps=2)
    assert stage == "sr"
    assert read_export(second, "sr")["lr"]["stage"] == "lr"
    expected, before, after = _losses(whole), _losses(first), _losses(second)
    for tag in LOSSES:
        assert before[tag] | after[tag] == expected[tag], tag


def _sample_file(folder: Path, images: Path, **tune) -> Path:
    sample = yaml.safe_load((TINY / "tiny_default.yaml").read_text())
    sample["images"] = {plane: str(images) for plane in ("xy", "xz", "yz")}
    for stage, values in tune.items():
        sample["tune"][stage].update(values)
    path = folder / "sample.yaml"
    path.write_text(yaml.safe_dump(sample), encoding="utf-8")
    return path


def test_a_mismatched_sr_stops_before_lr_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = _sample_file(tmp_path, TINY / "default", sr={"data": {"lo_res_size": 8}})
    with pytest.raises(ValueError, match="data.lo_res_size 16 \\(LR\\) != 8"):
        train.main([str(sample), "--run-dir", str(tmp_path / "run" / "x")])
    assert not (tmp_path / "run").exists()


def test_corruption_replaces_blocks_and_reports_the_level():
    low = torch.zeros(4, 2, 8, 8, 8)
    low[:, 0] = 1
    generator = torch.Generator().manual_seed(0)
    corrupted, level = corrupt_coarse(low, 1.0, 1.0, generator)
    changed = (corrupted != low).any(1)
    assert changed.any()
    torch.testing.assert_close(level, changed.float().mean((1, 2, 3)))
    same, zero = corrupt_coarse(low, 0.0, 1.0, generator)
    assert torch.equal(same, low) and not zero.any()


def test_a_height_conditioned_sr_stage_refines_at_a_depth(tiny_anode_sr_run):
    api = SuperResolutionAPI(tiny_anode_sr_run, "cpu")
    assert api.config.conditioning.height_enabled
    low = api.lr.generate(size=16, seed=0, height_origin=16.0)
    high = api.refine(low, seed=0, height_origin=16.0)
    assert high.shape == (32, 32, 32)
    with pytest.raises(ValueError, match="outside the source image height"):
        api.refine(low, seed=0, height_origin=48.0)
