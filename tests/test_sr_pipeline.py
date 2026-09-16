import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import run_predict
import run_train_2nd
from src.build.trainer import build_trainer
from src.config import load_train_config, load_yaml, save_yaml
from src.storage import load_volume
from src.train.run import run_train
from src.train.sr_run import file_hash, refresh_bank


@pytest.mark.parametrize("scale", [1.5, 4])
def test_stage1_to_sr_training_resume_and_cli_prediction(tmp_path, scale):
    torch.set_num_threads(1)
    cfg = load_train_config("config/train/low_res.yaml")
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
    sr_cfg = load_train_config("config/train/sr.yaml", "sr")
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
        checkpoint_every_steps=1,
    )
    sr_cfg["lr_bank"]["samples_per_domain"] = 1
    sr_cfg["lr_bank"]["refresh_every_steps"] = 1
    sr_config = tmp_path / "sr.yaml"
    data_file = tmp_path / "data.yaml"
    save_yaml(data_file, sr_cfg["data"])
    save_yaml(sr_config, {**sr_cfg, "data": str(data_file)})
    sr_dir = tmp_path / "sr"
    run_train_2nd.main(
        [
            "--config",
            str(sr_config),
            "--base-weights",
            str(base_dir),
            "--device",
            "cpu",
            "--run-dir",
            str(sr_dir),
        ]
    )
    stored = load_yaml(sr_dir / "config.yaml")
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
    assert Path(first["config"]["source"]["bank"]).is_file()
    out = tmp_path / "result.tiff"
    run_predict.main(
        [
            "--weights",
            str(base_dir),
            "--sr-weights",
            str(resumed / "weights/model.pt"),
            "--output",
            str(out),
            "--device",
            "cpu",
        ]
    )
    high_size = int(8 * scale)
    assert load_volume(out).shape == (high_size,) * 3
    assert load_volume(tmp_path / "result_lr.tiff").shape == (8, 8, 8)
    assert (
        json.loads(out.with_suffix(".json").read_text())["source_pixels_per_hr_voxel"]
        == 16 / high_size
    )
    out2 = tmp_path / "from_input.tiff"
    run_predict.main(
        [
            "--input",
            str(tmp_path / "result_lr_probs.pt"),
            "--sr-weights",
            str(resumed / "weights/model.pt"),
            "--output",
            str(out2),
            "--device",
            "cpu",
        ]
    )
    assert torch.equal(load_volume(out), load_volume(out2))


def test_bank_refresh_saves_new_bank_without_overwriting_resume_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "generator.pt"
    source.write_bytes(b"frozen weights")
    source_config = tmp_path / "train.yaml"
    source_config.write_text("stage: low_res\n", encoding="utf-8")
    old_bank = tmp_path / "lr_bank.pt"
    old_bank.write_bytes(b"original bank")
    cfg = {
        "data": {},
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
    )
    monkeypatch.setattr(
        "src.train.sr_run.load_generator",
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
        tmp_path / "lr_bank_step_00000002.pt"
    )
    source_config.write_text("stage: sr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        refresh_bank(trainer, tmp_path)
