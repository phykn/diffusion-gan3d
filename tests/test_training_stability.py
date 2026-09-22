import copy
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from src.build.model import build_models
from src.build.trainer import build_trainer
from src.config.files import save_yaml
from src.config.train import load_train_config
from src.data.slice import TripletBatch
from src.model.critic import ConnectivityCritic2D
from src.train.run.low_res import run_low_res_train
from src.train.state import resume_training, save_training
from src.train.step import step_optimizer


def small_config(tmp_path):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for index in range(4):
        labels = ((np.indices((12, 12)).sum(0) + index) % 3).astype(np.uint8)
        Image.fromarray(labels).save(images / f"{index}.png")
    cfg = load_train_config("tests/fixtures/config/train/low_res.yaml")
    cfg["data"].update(
        domains={0: {p: [str(images)] for p in ("xy", "xz", "yz")}},
        num_phases=3,
        crop_size=8,
        lo_res_size=8,
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"].update(channels=[4, 8], plane_groups=[["xy"], ["xz", "yz"]])
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["diffusion"].update(num_steps=2, time_embedding="scaled")
    cfg["conditioning"]["anchor"].update(probability=1.0, ramp_steps=0)
    cfg["loss"].update(r1_weight=0.01, r2_weight=0.01, r1_every_steps=2)
    cfg["train"].update(
        total_steps=5, real_batch_size=2, slice_pairs_per_plane=2, mixed_precision=False
    )
    return cfg


def test_lr_checkpoint_restores_training_state_without_rng(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    trainer.step(0, transition=0)
    path = tmp_path / "last.pt"
    save_training(path, trainer)
    payload = torch.load(path, weights_only=True)
    assert payload["format"] == "diffusion-gan3d.lr.train"
    assert (
        not {"streams", "torch_rng", "cuda_rng", "numpy_rng", "python_rng"}
        & payload.keys()
    )
    weights = copy.deepcopy(trainer.denoiser.state_dict())
    critic_weights = copy.deepcopy(trainer.critics.state_dict())
    restored = build_trainer(cfg, torch.device("cpu"))
    restored.scaler.update = Mock(wraps=restored.scaler.update)
    rng = torch.get_rng_state().clone()
    resume_training(restored, payload)
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.updates == trainer.updates
    assert restored.completed_steps == 1
    torch.testing.assert_close(
        restored.denoiser_optim.state_dict(), trainer.denoiser_optim.state_dict()
    )
    for group in trainer.critic_optims:
        torch.testing.assert_close(
            restored.critic_optims[group].state_dict(),
            trainer.critic_optims[group].state_dict(),
        )
    for name, value in weights.items():
        torch.testing.assert_close(
            restored.denoiser.state_dict()[name], value, rtol=0, atol=0
        )
    for name, value in critic_weights.items():
        torch.testing.assert_close(
            restored.critics.state_dict()[name], value, rtol=0, atol=0
        )
    metrics = [restored.step(i, transition=0) for i in range(1, 4)]
    assert restored.scaler.update.call_count == 3
    assert restored.completed_steps == 4
    assert any(key.startswith("r2/") for row in metrics for key in row.diagnostics)
    assert any(
        key.startswith("input_gradient/") for row in metrics for key in row.diagnostics
    )


def test_lr_resume_rejects_changed_images_or_training_contract(tmp_path):
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    path = tmp_path / "last.pt"
    save_training(path, trainer)
    payload = torch.load(path, weights_only=True)
    changed = copy.deepcopy(cfg)
    changed["loss"]["r2_weight"] = 0.2
    with pytest.raises(ValueError, match="saved settings"):
        resume_training(build_trainer(changed, torch.device("cpu")), payload)
    Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(
        tmp_path / "images" / "0.png"
    )
    with pytest.raises(ValueError, match="images changed"):
        resume_training(build_trainer(cfg, torch.device("cpu")), payload)


def test_lr_cli_resumes_progress_without_training_seed(tmp_path):
    cfg = small_config(tmp_path)
    cfg["train"]["weights_every_steps"] = 1
    preset = tmp_path / "recipe.yaml"
    save_yaml(preset, cfg)

    def run(*args):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "run_train_1st.py",
                "--device",
                "cpu",
                *map(str, args),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    run("--config", preset, "--steps", 2, "--run-dir", tmp_path / "first")
    run(
        "--resume",
        tmp_path / "first/checkpoints/last.pt",
        "--steps",
        4,
        "--run-dir",
        tmp_path / "resumed",
    )
    resumed = torch.load(tmp_path / "resumed/checkpoints/last.pt", weights_only=True)
    assert resumed["step"] == 4
    assert "seed" not in resumed["config"]["train"]
    assert "torch_rng" not in resumed
    assert resumed["updates"]["generator"] == 4


def test_lr_resume_after_moving_files_preserves_holdouts_and_hash_checks(tmp_path):
    torch.set_num_threads(1)
    original = tmp_path / "original"
    original.mkdir()
    cfg = small_config(original)
    cfg["data"]["split"] = {
        "validation_files": [str(original / "images/3.png")],
        "validation_regions": {str(original / "images/0.png"): [0, 0, 1, 1]},
    }
    preset = original / "recipe.yaml"
    save_yaml(preset, cfg)
    run_low_res_train(config=preset, steps=1, run_dir=original / "run")
    moved = tmp_path / "moved"
    original.rename(moved)
    checkpoint = moved / "run/checkpoints/last.pt"
    run_low_res_train(
        resume=checkpoint,
        steps=2,
        run_dir=moved / "resumed",
        path_map=[(str(original), str(moved))],
    )
    saved = torch.load(moved / "resumed/checkpoints/last.pt", weights_only=True)
    assert saved["step"] == 2
    assert len(saved["data_fingerprint"]) == 3
    assert saved["config"]["data"]["split"]["validation_files"] == [
        str(moved / "images/3.png")
    ]
    assert list(saved["config"]["data"]["split"]["validation_regions"]) == [
        str(moved / "images/0.png")
    ]
    assert saved["path_maps"]
    run_low_res_train(
        resume=moved / "resumed/checkpoints/last.pt", steps=3, run_dir=moved / "again"
    )
    Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(moved / "images/0.png")
    with pytest.raises(ValueError, match="images changed"):
        run_low_res_train(
            resume=checkpoint,
            steps=2,
            run_dir=moved / "invalid",
            path_map=[(str(original), str(moved))],
        )


def test_connectivity_preserves_height_order():
    torch.manual_seed(3)
    old = ConnectivityCritic2D(2, [4, 8], 8, 1)
    directed = ConnectivityCritic2D(2, [4, 8], 8, 1, directed_axis=0)
    directed.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(3, 3, 2, 8, 8)
    axes = torch.tensor([0, 1, 2])
    gaps, domains = torch.ones(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)
    a, b = directed(x, axes, gaps, domains), directed(x.flip(1), axes, gaps, domains)
    assert a.logits_local.shape == (3, 4, 4)
    assert not torch.equal(a.logits_local[0], b.logits_local[0])
    assert torch.equal(a.logits_local[1:], b.logits_local[1:])
    assert torch.equal(
        old(x, axes, gaps, domains).logits_local,
        old(x.flip(1), axes, gaps, domains).logits_local,
    )


@pytest.mark.parametrize("height", [False, True])
def test_time_scaling_is_shared_without_changing_weight_shapes(tmp_path, height):
    cfg = small_config(tmp_path)
    cfg["conditioning"]["height_enabled"] = height
    generator, critics, connectivity = build_models(cfg)
    assert generator.time_scale == 500
    assert all(critic.time_scale == 500 for critic in critics.values())
    assert connectivity.directed_axis == (0 if height else None)
    cfg["model"]["diffusion"]["time_embedding"] = "index"
    unscaled, _, _ = build_models(cfg)
    unscaled.load_state_dict(generator.state_dict(), strict=True)
    assert unscaled.time_scale == 1


def test_nonfinite_gradient_does_not_update_fp32_parameters():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1)
    parameter.grad = torch.tensor(float("inf"))
    diagnostics = {}
    with pytest.raises(FloatingPointError, match="gradient"):
        step_optimizer(optimizer, None, diagnostics, "generator")
    assert parameter.item() == 1
    assert diagnostics["skipped/generator"] == 1


def test_connectivity_regularization_counts_its_own_updates(tmp_path):
    trainer = build_trainer(small_config(tmp_path), torch.device("cpu"))
    values = torch.randn(2, 3, 3, 8, 8)
    fake = TripletBatch(
        values,
        torch.tensor([0, 1]),
        torch.ones(2, dtype=torch.long),
        torch.ones(2, dtype=torch.long),
    )
    domains = torch.zeros(2, dtype=torch.long)
    trainer.update_connectivity_critic(values + 0.1, fake, 15, domains)
    assert trainer.diagnostics["regularization/connectivity"] == 0
    trainer.update_connectivity_critic(values + 0.1, fake, 101, domains)
    assert trainer.diagnostics["regularization/connectivity"] == 1
    assert trainer.updates["connectivity"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_skip_is_reported_without_updating_parameters():
    parameter = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    optimizer = torch.optim.SGD([parameter], lr=1)
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(parameter * float("inf")).backward()
    diagnostics = {}
    assert not step_optimizer(optimizer, scaler, diagnostics, "generator")
    scaler.update()
    assert parameter.item() == 1
    assert diagnostics["skipped/generator"] == 1
