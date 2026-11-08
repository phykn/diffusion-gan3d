"""SR stage: tags, exports, resume without the LR run, SR source checks."""

import shutil
from pathlib import Path

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train
from src.api import SuperResolutionAPI
from src.config.sample import build_stage_configs, load_sample
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


@pytest.fixture
def lr_copy(tiny_lr_run, tmp_path) -> Path:
    """A private copy of the tiny LR run that a test may delete."""
    return Path(shutil.copytree(tiny_lr_run, tmp_path / "copy-lr"))


def test_a_copied_sr_export_refines_alone(tiny_configs, lr_copy, tmp_path):
    sr = train_stage(tiny_configs["sr"], "cpu", tmp_path / "tiny-sr", lr_dir=lr_copy)
    alone = tmp_path / "alone" / "generator.pt"
    alone.parent.mkdir()
    shutil.copy(sr / "generator.pt", alone)
    shutil.rmtree(lr_copy)
    shutil.rmtree(sr)
    api = SuperResolutionAPI(alone, "cpu")
    low = api.lr.generate(size=32, seed=0)
    assert api.refine(low, seed=0).shape == (64, 64, 64)
    tile, overlap = api.default_tiling()
    tiled = api.refine(low, seed=0, tile_size=tile)
    assert tiled.shape == (64, 64, 64)
    assert overlap % api.scale_factor == 0


def test_a_moved_sr_checkpoint_resumes_without_the_lr_run(
    tiny_configs, lr_copy, tmp_path
):
    config = update(tiny_configs["sr"], "train.total_steps", 2)
    sr = train_stage(config, "cpu", tmp_path / "tiny-sr", lr_dir=lr_copy)
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("step_00000002.pt", "step_00000002.complete"):
        shutil.move(sr / "checkpoints" / name, moved / name)
    shutil.rmtree(lr_copy)
    shutil.rmtree(sr)
    stage, resumed = resume_stage(moved, "cpu", tmp_path / "resumed", steps=3)
    assert stage == "sr"
    assert (resumed / ".done").is_file()
    assert read_export(resumed, "sr")["lr"]["stage"] == "lr"


def test_sr_resume_matches_an_unbroken_run(tiny_configs, lr_copy, tmp_path):
    # The bank refreshes every step, so the bank must resume exactly too.
    config = update(tiny_configs["sr"], "train.archive_every_steps", 2)
    config = update(config, "train.structure_every_steps", 0)
    whole = train_stage(
        update(config, "train.total_steps", 4), "cpu", tmp_path / "whole", lr_copy
    )
    first = train_stage(
        update(config, "train.total_steps", 2), "cpu", tmp_path / "first", lr_copy
    )
    _, second = resume_stage(first, "cpu", tmp_path / "second", steps=4)
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


def test_sr_reads_its_own_images_after_the_lr_folder_moves(tmp_path):
    images = tmp_path / "images"
    shutil.copytree(TINY / "default", images)
    lr_configs = build_stage_configs(load_sample(_sample_file(tmp_path, images)))
    lr = train_stage(lr_configs["lr"], "cpu", tmp_path / "run-lr")
    moved = tmp_path / "moved-images"
    images.rename(moved)
    configs = build_stage_configs(load_sample(_sample_file(tmp_path, moved)))
    sr = train_stage(configs["sr"], "cpu", tmp_path / "run-sr", lr_dir=lr)
    assert (sr / ".done").is_file()


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
