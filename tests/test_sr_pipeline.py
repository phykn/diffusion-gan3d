import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

import run_predict
import run_sr_train
from src.build.trainer import build_trainer
from src.config import load_train_config, load_yaml, save_yaml
from src.storage import load_volume
from src.train.run import run_train


@pytest.mark.parametrize("legacy,scale", [(False, 1.5), (False, 4), (True, 1.5)])
def test_stage1_to_sr_training_resume_and_cli_prediction(tmp_path, legacy, scale):
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
        domains={0: {0: [str(images)], 1: [str(images)], 2: [str(images)]}},
    )
    if legacy:
        cfg["data"]["scale_factor"] = scale
        cfg["data"].pop("thickness_axis", None)
        cfg["augmentation"] = {"mode": "anisotropic", "probability": 0.5}
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
    if legacy:
        sr_cfg["augmentation"] = copy.deepcopy(cfg["augmentation"])
    sr_cfg["data"].pop("scale_factor", None)
    sr_images = tmp_path / "sr_images"
    sr_images.mkdir()
    (sr_images / "sample.png").write_bytes((images / "sample.png").read_bytes())
    sr_cfg["data"]["domains"] = {
        0: {plane: [str(sr_images)] for plane in ("xy", "xz", "yz")}
    }
    sr_cfg["model"]["generator"].update(channels=4, blocks=1, scale_factor=scale)
    sr_cfg["model"]["critic"]["channels"] = [4, 8]
    sr_cfg["train"].update(
        slices_per_plane=2,
        critic_updates_per_step=1,
        mixed_precision=False,
        total_steps=1,
        checkpoint_every_steps=1,
    )
    sr_cfg["lr_bank"]["samples_per_domain"] = 1
    sr_config = tmp_path / "sr.yaml"
    if legacy:
        preset = legacy_sr_config(sr_cfg)
        del preset["data"]
        save_yaml(sr_config, preset)
    else:
        data_file = tmp_path / "data.yaml"
        save_yaml(data_file, sr_cfg["data"])
        save_yaml(sr_config, {**sr_cfg, "data": str(data_file)})
    sr_dir = tmp_path / "sr"
    run_sr_train.main(
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
    expected_data = cfg["data"] if legacy else sr_cfg["data"]
    assert stored["data"]["domains"] == expected_data["domains"]
    assert "scale_factor" not in stored["data"]
    assert stored["model"]["generator"]["scale_factor"] == scale
    assert stored["data"]["crop_size"] == 16
    assert stored["data"]["lo_res_size"] == 8
    first = torch.load(sr_dir / "checkpoints/last.pt", weights_only=True)
    assert first["step"] == 1
    if legacy:
        first["config"] = legacy_sr_config(first["config"])
        torch.save(first, sr_dir / "checkpoints/last.pt")
    resumed = tmp_path / "resumed"
    run_sr_train.main(
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
    assert second["config"]["source"] == first["config"]["source"]
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
            str(tmp_path / "result_lr.tiff"),
            "--sr-weights",
            str(resumed / "weights/model.pt"),
            "--output",
            str(out2),
            "--device",
            "cpu",
        ]
    )
    assert torch.equal(load_volume(out), load_volume(out2))


def legacy_sr_config(cfg):
    data = copy.deepcopy(cfg["data"])
    data["num_phase"] = data.pop("num_phases")
    model = copy.deepcopy(cfg["model"]["generator"])
    data["scale_factor"] = model.pop("scale_factor")
    data.update(
        augment=cfg["augmentation"]["mode"],
        augment_prob=cfg["augmentation"]["probability"],
    )
    train_names = {
        "steps": "total_steps",
        "amp": "mixed_precision",
        "batch_size": "volume_batch_size",
        "slices_per_axis": "slices_per_plane",
        "critic_steps": "critic_updates_per_step",
        "save_every": "checkpoint_every_steps",
    }
    loss_names = {
        "gradient_penalty": "gradient_penalty_weight",
        "lr_weight": "downsample_consistency_weight",
        "lr_tolerance": "downsample_mse_tolerance",
        "temperature": "downsample_temperature",
    }
    optim = copy.deepcopy(cfg["optim"])
    optim["betas"] = optim.pop("adam_betas")
    old = {
        "data": data,
        "model": model,
        "critic": cfg["model"]["critic"],
        "optim": optim,
        "train": {
            **{old: cfg["train"][new] for old, new in train_names.items()},
            **{old: cfg["loss"][new] for old, new in loss_names.items()},
            "bank_size": cfg["lr_bank"]["samples_per_domain"],
        },
    }
    if "source" in cfg:
        old["source"] = cfg["source"]
    return old
