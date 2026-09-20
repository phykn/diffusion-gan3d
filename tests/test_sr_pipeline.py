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

import run_train_2nd
from src.build.trainer import build_trainer
from src.config.files import load_yaml, save_yaml
from src.config.train import load_train_config
from src.storage import load_volume
from src.train.run.bank import file_hash, refresh_bank, save_bank
from src.train.run.loop import run_train


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
    assert trainer.patch_size == 8
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
    data_file = tmp_path / "data.yaml"
    save_yaml(data_file, sr_cfg["data"])
    save_yaml(sr_config, {**sr_cfg, "data": str(data_file)})
    monkeypatch.setattr("src.train.run.sr.PROJECT_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    source_args = ["--base-weights", str(base_dir)] if source_mode == "cli" else []
    run_train_2nd.main(
        [
            "--config",
            str(sr_config),
            *source_args,
            "--device",
            "cpu",
        ]
    )
    (sr_dir,) = (tmp_path / "run").iterdir()
    assert sr_dir.name.endswith("_sr_refine")
    common_files = {
        "train.yaml",
        "data_manifest.json",
        "generator.pt",
        "metrics.jsonl",
        "checkpoints",
        "tensorboard",
    }
    for directory in (base_dir, sr_dir):
        assert common_files <= {path.name for path in directory.iterdir()}
        assert (directory / "checkpoints/last.pt").is_file()
        assert list(directory.glob("critic_*.pt"))
        events = EventAccumulator(str(directory / "tensorboard")).Reload()
        assert events.Scalars("loss/generator")[0].step == 1
    archive = sr_dir / "checkpoints/step_00000001/generator.pt"
    archived = torch.load(archive, weights_only=True)
    assert archived["format"] == "diffusion-gan3d.sr"
    assert archived["step"] == 1
    stored = load_yaml(sr_dir / "train.yaml")
    assert stored["source"]["weights"] == str((base_dir / "generator.pt").resolve())
    assert Path(stored["source"]["bank"]) == sr_dir / "lr_bank/step_00000000.pt"
    assert not list(sr_dir.glob("lr_bank*.pt"))
    expected_data = sr_cfg["data"]
    assert stored["data"]["domains"] == expected_data["domains"]
    assert "scale_factor" not in stored["data"]
    assert stored["data"]["hi_res_size"] == int(8 * scale)
    assert stored["data"]["crop_size"] == 16
    assert stored["data"]["lo_res_size"] == 8
    first = torch.load(sr_dir / "checkpoints/last.pt", weights_only=True)
    assert first["step"] == 1
    resumed = tmp_path / "resumed"
    run_train_2nd.main(
        [
            "--resume",
            str(sr_dir / "checkpoints/last.pt"),
            "--steps",
            "2",
            "--run-dir",
            str(resumed),
            "--device",
            "cpu",
        ]
    )
    second = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
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
    assert load_yaml(resumed / "train.yaml")["source"] == second["config"]["source"]
    check_hr = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/06_check_hr.py")
    )["main"]
    out = tmp_path / "result"
    check_hr(
        [
            "--lr-weight",
            str(base_dir),
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
            str(resumed / "checkpoints/step_00000002"),
            "--out",
            str(out2),
            "--no-view",
            "--device",
            "cpu",
        ]
    )
    assert torch.equal(load_volume(out / "hr.tiff"), load_volume(out2 / "hr.tiff"))


def test_bank_refresh_saves_new_bank_without_overwriting_resume_source(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "generator.pt"
    source.write_bytes(b"frozen weights")
    source_config = source_dir / "train.yaml"
    source_config.write_text("stage: low_res\n", encoding="utf-8")
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
        },
        "lr_bank": {"refresh_every_steps": 2, "guidance": 1},
    }
    trainer = SimpleNamespace(
        cfg=cfg,
        completed_steps=2,
        device=torch.device("cpu"),
        bank={0: torch.full((2, 2, 8, 8, 8), 0.5)},
        bank_origins=None,
        bank_extents=None,
        bank_conditions=None,
    )
    monkeypatch.setattr(
        "src.train.run.bank.load_generator",
        lambda *args: SimpleNamespace(
            generate_probs=lambda **kwargs: torch.stack(
                (torch.ones(8, 8, 8), torch.zeros(8, 8, 8))
            )
        ),
    )
    refresh_bank(trainer, tmp_path)
    assert old_bank.read_bytes() == b"original bank"
    assert trainer.bank[0][0, 0].eq(1).all()
    assert trainer.bank[0][1].eq(0.5).all()
    assert cfg["source"]["bank_sha256"] == file_hash(
        tmp_path / "lr_bank/step_00000002.pt"
    )
    source_config.write_text("stage: sr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        refresh_bank(trainer, tmp_path)


def test_bank_snapshot_cannot_overwrite_a_published_step(tmp_path):
    source = save_bank(tmp_path, 0, {"volumes": torch.ones(2)})
    path = Path(source["bank"])
    digest = source["bank_sha256"]
    with pytest.raises(FileExistsError):
        save_bank(tmp_path, 0, {"volumes": torch.zeros(2)})
    assert file_hash(path) == digest == source["bank_sha256"]
    assert torch.load(path, weights_only=True)["volumes"].eq(1).all()


def test_sr_default_config_comes_from_project_root(monkeypatch):
    from src.config.files import PROJECT_ROOT
    from src.train.run.sr import run_sr_train

    def read_config(path, stage, **kwargs):
        assert path == PROJECT_ROOT / "config/train/sr.yaml"
        assert path.is_file()
        assert stage == "sr"
        raise RuntimeError("default config reached")

    monkeypatch.setattr("src.train.run.sr.load_train_config", read_config)
    with pytest.raises(RuntimeError, match="default config reached"):
        run_sr_train()
