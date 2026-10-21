import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.build.trainer import build_trainer
from src.config.files import load_yaml, save_yaml
from src.config.generation import load_generation_settings
from src.config.train import load_train_config
from src.storage import file_hash, load_volume
from src.train.checkpoint import resolve_checkpoint
from src.train.run.bank import publish_bank, refresh_bank
from src.train.run.loop import run_train
from src.train.run.low_res import run_low_res_train
from src.train.run.sr import run_sr_train


@pytest.mark.parametrize(
    "scale,source_mode", [(1.5, "config_folder"), (4, "config_file"), (1.5, "cli")]
)
def test_stage1_to_sr_training_resume_and_cli_prediction(
    tmp_path, scale, source_mode, monkeypatch
):
    torch.set_num_threads(1)
    cfg = load_train_config("tests/fixtures/config/train/low_res.yaml")
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray((np.indices((20, 20)).sum(0) % 3).astype(np.uint8)).save(
        images / "sample.png"
    )
    cfg["data"].update(
        crop_size=16,
        lo_res_size=8,
        num_phases=3,
        domains={0: {p: [str(images)] for p in ("xy", "xz", "yz")}},
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"]["channels"] = [4, 8]
    cfg["loss"]["r1_weight"] = 0.0
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["conditioning"]["anchor"].update(probability=0.0, start_step=0)
    cfg["loss"]["connectivity"]["adversarial_weight"] = 0.0
    cfg["train"].update(
        total_steps=1, mixed_precision=False, slice_pairs_per_plane=2, real_batch_size=2
    )
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    save_yaml(base_dir / "train.yaml", cfg)
    trainer = build_trainer(cfg, torch.device("cpu"))
    assert trainer.settings.patch_size == 8
    run_train(trainer, steps=1, save_every=1, run_dir=base_dir)
    sr_cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    sr_cfg["nickname"] = "refine"
    sr_cfg["data"] = copy.deepcopy(cfg["data"])
    sr_images = tmp_path / "sr_images"
    sr_images.mkdir()
    (sr_images / "sample.png").write_bytes((images / "sample.png").read_bytes())
    sr_cfg["data"]["domains"] = {
        0: {plane: [str(sr_images)] for plane in ("xy", "xz", "yz")}
    }
    sr_cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    sr_cfg["model"]["diffusion"]["num_steps"] = 2
    sr_cfg["model"]["gradient_checkpointing"] = False
    sr_cfg["data"]["hi_res_size"] = int(8 * scale)
    sr_cfg["model"]["critic"]["channels"] = [4, 8]
    sr_cfg["train"].update(
        real_batch_size=2,
        slice_pairs_per_plane=2,
        mixed_precision=False,
        total_steps=1,
        weights_every_steps=1,
        archive_every_steps=1,
    )
    sr_cfg["lr_bank"]["samples_per_domain"] = 1
    sr_cfg["lr_bank"]["refresh_every_steps"] = 1
    sr_cfg["source"] = {
        "weights": {
            "config_folder": "base",
            "config_file": "base/generator.pt",
            "cli": "missing_override",
        }[source_mode]
    }
    sr_config = tmp_path / "sr.yaml"
    save_yaml(sr_config, sr_cfg)
    monkeypatch.setattr("src.train.run.sr.PROJECT_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run_sr_train(
        "cpu",
        config=sr_config,
        base_weights=base_dir if source_mode == "cli" else None,
    )
    (sr_dir,) = (tmp_path / "run").iterdir()
    assert sr_dir.name.endswith("_sr_refine")
    common_files = {
        "train.yaml",
        "generator.pt",
        "checkpoints",
        "tensorboard",
    }
    for directory in (base_dir, sr_dir):
        assert common_files <= {path.name for path in directory.iterdir()}
        assert resolve_checkpoint(directory).name == "step_00000001.pt"
        assert (directory / "checkpoints/generator_00000001.pt").is_file()
        assert not (directory / "data_manifest.json").exists()
        assert not (directory / "metrics.jsonl").exists()
        assert list(directory.glob("critic_*.pt"))
        events = EventAccumulator(str(directory / "tensorboard")).Reload()
        assert events.Scalars("loss/generator")[0].step == 1
    archive = resolve_checkpoint(sr_dir)
    archived = torch.load(archive, weights_only=True)
    assert archived["format"] == "diffusion-gan3d.sr.train"
    assert archived["step"] == 1
    stored = load_yaml(sr_dir / "train.yaml")
    expected = load_yaml(sr_config)
    if source_mode == "cli":
        expected["source"]["weights"] = str(base_dir)
    assert stored == expected
    source = archived["config"]["source"]
    assert source["weights"] == str((base_dir / "generator.pt").resolve())
    assert Path(source["bank"]) == sr_dir / "lr_bank/step_00000000.pt"
    assert not list(sr_dir.glob("lr_bank*.pt"))
    expected_data = sr_cfg["data"]
    resolved = archived["config"]["data"]
    assert resolved["domains"] == expected_data["domains"]
    assert "scale_factor" not in resolved
    assert resolved["hi_res_size"] == int(8 * scale)
    assert resolved["crop_size"] == 16
    assert resolved["lo_res_size"] == 8
    first = torch.load(resolve_checkpoint(sr_dir), weights_only=True)
    assert first["step"] == 1
    resumed = tmp_path / "resumed"
    run_sr_train("cpu", resume=sr_dir, steps=2, run_dir=resumed)
    second = torch.load(resolve_checkpoint(resumed), weights_only=True)
    assert second["step"] == 2
    assert (
        second["config"]["source"]["weights_sha256"]
        == first["config"]["source"]["weights_sha256"]
    )
    assert second["config"]["source"]["bank"] != first["config"]["source"]["bank"]
    assert Path(second["config"]["source"]["bank"]).is_file()
    assert Path(second["config"]["source"]["bank"]).parent == resumed / "lr_bank"
    assert Path(first["config"]["source"]["bank"]).is_file()
    assert Path(archived["config"]["source"]["bank"]).is_file()
    # A resume saves the resolved checkpoint settings.
    assert load_yaml(resumed / "train.yaml") == {
        **first["config"],
        "train": {**first["config"]["train"], "total_steps": 2},
    }
    check_hr = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/06_check_hr.py")
    )["main"]
    out = tmp_path / "result"
    check_hr(
        [
            "--weight",
            str(resumed),
            "--out",
            str(out),
            "--no-view",
            "--device",
            "cpu",
        ]
    )
    high_size = int(8 * scale)
    assert load_volume(out / "hr.tiff").shape == (high_size,) * 3
    assert load_volume(out / "lr.tiff").shape == (8, 8, 8)
    assert (
        json.loads((out / "report.json").read_text())["source_pixels_per_hr_voxel"]
        == 16 / high_size
    )
    out2 = tmp_path / "from_input"
    check_hr(
        [
            "--input",
            str(out / "lr_probs.pt"),
            "--weight",
            str(resumed / "checkpoints/generator_00000002.pt"),
            "--out",
            str(out2),
            "--no-view",
            "--device",
            "cpu",
        ]
    )
    assert torch.equal(load_volume(out / "hr.tiff"), load_volume(out2 / "hr.tiff"))
    report = json.loads((out / "report.json").read_text())
    assert report["lr_source_verified"] is True
    assert report["lr_guidance"] == load_generation_settings().guidance
    assert json.loads((out2 / "report.json").read_text())["lr_guidance"] is None
    with (base_dir / "train.yaml").open("a") as config_file:
        config_file.write("\n# changed after SR training\n")
    with pytest.raises(ValueError, match="configuration changed"):
        check_hr(["--weight", str(resumed), "--no-view", "--device", "cpu"])


def test_bank_refresh_saves_new_bank_without_overwriting_resume_source(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "generator.pt"
    source.write_bytes(b"frozen weights")
    source_config = source_dir / "train.yaml"
    save_yaml(source_config, load_yaml("tests/fixtures/config/train/low_res.yaml"))
    old_bank = tmp_path / "lr_bank/step_00000000.pt"
    old_bank.parent.mkdir()
    old_bank.write_bytes(b"original bank")
    cfg = {
        "data": {},
        "conditioning": {"height_enabled": False},
        "source": {
            "weights": str(source),
            "weights_sha256": file_hash(source),
            "config_sha256": file_hash(source_config),
            "bank": str(old_bank),
            "bank_sha256": file_hash(old_bank),
        },
        "lr_bank": {"refresh_every_steps": 2, "guidance": 1},
    }
    bank = {
        "format": "diffusion-gan3d.lr-bank",
        "volumes": {0: torch.full((2, 2, 8, 8, 8), 0.5)},
        "height_origins": None,
        "height_extents": None,
        "conditions": None,
    }
    monkeypatch.setattr(
        "src.train.run.bank.build_generator",
        lambda *args: SimpleNamespace(
            generate_probs=lambda **kwargs: torch.stack(
                (torch.ones(8, 8, 8), torch.zeros(8, 8, 8))
            )
        ),
    )
    original_source = dict(cfg["source"])
    refresh_bank(bank, cfg, 2, torch.device("cpu"))
    assert cfg["source"] == original_source
    assert not (tmp_path / "lr_bank/step_00000002.pt").exists()
    publish_bank(bank, cfg, tmp_path, 2)
    published = torch.load(cfg["source"]["bank"], weights_only=True)
    assert published["source"] == {
        key: value
        for key, value in original_source.items()
        if key not in {"bank", "bank_sha256"}
    }
    assert old_bank.read_bytes() == b"original bank"
    assert bank["volumes"][0][0, 0].eq(1).all()
    assert bank["volumes"][0][1].eq(0.5).all()
    assert cfg["source"]["bank_sha256"] == file_hash(
        tmp_path / "lr_bank/step_00000002.pt"
    )
    source_config.write_text("stage: sr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        refresh_bank(bank, cfg, 4, torch.device("cpu"))


def test_bank_snapshot_cannot_overwrite_a_published_step(tmp_path):
    cfg = {"data": {}, "source": {}}
    publish_bank({"volumes": torch.ones(2)}, cfg, tmp_path, 0)
    source = dict(cfg["source"])
    path = Path(source["bank"])
    with pytest.raises(FileExistsError):
        publish_bank({"volumes": torch.zeros(2)}, cfg, tmp_path, 0)
    assert file_hash(path) == source["bank_sha256"]
    assert torch.load(path, weights_only=True)["volumes"].eq(1).all()


def test_new_runs_need_a_training_config():
    for runner in (run_low_res_train, run_sr_train):
        with pytest.raises(ValueError, match="needs a training config"):
            runner()
