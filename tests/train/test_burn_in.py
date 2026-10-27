import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.build.trainer import build_trainer
from src.config.files import load_yaml
from src.config.train import normalize_train_config
from src.plane import PLANES
from src.train.burn_in import BurnIn

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> dict:
    folder = tmp_path / "slices"
    folder.mkdir()
    Image.fromarray((np.indices((8, 8)).sum(0) % 3).astype(np.uint8)).save(
        folder / "0.png"
    )
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
        real_batch_size=2, slice_pairs_per_plane=2, mixed_precision=False
    )
    cfg["selection"] = {"enabled": False}
    return cfg


def _burn_in(steps=4):
    return BurnIn(
        steps=steps,
        generator_lr=2e-4,
        critic_lr=1e-4,
        betas=(0.5, 0.9),
        ema_decay=0.999,
        lr_scale=0.25,
        beta2=0.99,
        regularization_scale=0.1,
    )


def test_values_follow_a_cosine_from_start_to_end():
    burn_in = _burn_in()
    start, middle, end, after = (burn_in.values(step) for step in (0, 2, 4, 9))
    assert (start.generator_lr, start.critic_lr) == (2e-4, 1e-4)
    assert start.betas == (0.5, 0.9)
    assert start.regularization_scale == 1.0
    # Zero half-life: the first EMA update copies the weights.
    assert start.ema_decay == 0.0
    assert middle.generator_lr == pytest.approx(2e-4 * 0.625)
    assert middle.betas[1] == pytest.approx(0.945)
    assert middle.regularization_scale == pytest.approx(0.55)
    assert middle.ema_decay == pytest.approx(0.999**2)
    for values in (end, after):
        assert values.generator_lr == pytest.approx(5e-5)
        assert values.critic_lr == pytest.approx(2.5e-5)
        assert values.betas == pytest.approx((0.5, 0.99))
        assert values.regularization_scale == pytest.approx(0.1)
        assert values.ema_decay == pytest.approx(0.999)


def test_zero_steps_start_at_the_end_values():
    values = _burn_in(steps=0).values(0)
    assert values.generator_lr == pytest.approx(5e-5)
    assert values.ema_decay == pytest.approx(0.999)
    assert not math.isnan(_burn_in().progress(0))


def test_trainer_applies_the_schedule_each_step(tmp_path):
    torch.set_num_threads(1)
    cfg = _config(tmp_path)
    cfg["burn_in"] = {"steps": 2}
    cfg["loss"].update(r1_weight=0.5, r2_weight=0.2, r1_every_steps=1)
    trainer = build_trainer(cfg, torch.device("cpu"))
    optim = trainer.cfg["optim"]
    first = trainer.step(0, transition=0)
    assert first.diagnostics["burn_in/generator_lr"] == optim["generator_lr"]
    assert first.diagnostics["burn_in/ema_decay"] == 0.0
    # The EMA copied the generator at the first update.
    for ema, model in zip(
        trainer.ema_denoiser.parameters(), trainer.denoiser.parameters(), strict=True
    ):
        assert torch.equal(ema, model)
    trainer.step(2, transition=0)
    group = trainer.denoiser_optim.param_groups[0]
    assert group["lr"] == pytest.approx(optim["generator_lr"] * 0.25)
    assert group["betas"] == pytest.approx((optim["adam_betas"][0], 0.99))
    for optimizer in trainer.critic_optims.values():
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            optim["critic_lr"] * 0.25
        )
    assert trainer.regularization(next(iter(trainer.critic_optims))) == (
        True,
        pytest.approx(0.05),
        pytest.approx(0.02),
    )


def test_burn_in_settings_are_validated():
    assert normalize_train_config({})["burn_in"] == {
        "steps": 5000,
        "lr_scale": 0.25,
        "beta2": 0.99,
        "regularization_scale": 0.1,
    }
    assert normalize_train_config({}, "sr")["burn_in"]["steps"] == 2500
    for key, value in (
        ("steps", -1),
        ("steps", 1.5),
        ("lr_scale", -0.1),
        ("beta2", 1.0),
        ("regularization_scale", float("nan")),
    ):
        with pytest.raises(ValueError, match=f"burn_in.{key}"):
            normalize_train_config({"burn_in": {key: value}})
