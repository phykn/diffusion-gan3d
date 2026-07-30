import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile

from src.build import build_models
from src.misc import save_mapping
from src.train import save_model_weights
from src.train.config import (
    AnchorConfig,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    TrainingConfig,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename, expected",
    (
        ("03_check_anchor.py", "anchor_index=4"),
        ("04_check_anchor_all.py", "anchor_count=8"),
    ),
)
def test_anchor_check_script_runs_with_fixed_volume(
    filename: str,
    expected: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    volume_path = tmp_path / "volume_000.tif"
    volume = (np.indices((8, 8, 8)).sum(axis=0) % 3).astype(np.uint8)
    tifffile.imwrite(volume_path, volume, metadata={"axes": "ZYX"})

    cfg = _config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_mapping(run_dir / "train.yaml", cfg.as_dict())
    model, critics = build_models(cfg.data, cfg.model)
    weights = save_model_weights(run_dir, model, critics)

    module = _load_script(filename)
    monkeypatch.setattr(module, "VOLUME_PATH", volume_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [filename, "--weights", str(weights)],
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    module.main()
    plt.close("all")

    output = capsys.readouterr().out
    assert expected in output
    assert "accuracy=" in output


def _load_script(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(root: Path) -> TrainConfig:
    return TrainConfig(
        data=DataConfig(
            folder={axis: root / str(axis) for axis in (0, 1, 2)},
            crop_size=8,
            patch_size=8,
            num_phases=3,
            batch_size=2,
        ),
        model=ModelConfig(
            base_channels=4,
            channel_multipliers=(1, 2),
            embedding_channels=8,
            latent_channels=4,
            critic_channels=(4, 8),
            gradient_checkpointing=False,
        ),
        diffusion=DiffusionConfig(
            timesteps=1,
            beta_min=0.1,
            beta_max=2.0,
        ),
        anchor=AnchorConfig(
            probability=1.0,
            loss_weight=1.0,
        ),
        optim=OptimConfig(
            generator_lr=1e-3,
            critic_lr=1e-3,
            beta1=0.0,
            beta2=0.9,
            r1_gamma=0.0,
            r1_interval=2,
        ),
        train=TrainingConfig(
            steps=1,
            volume_batch_size=1,
            slices_per_axis=2,
            mixed_precision=False,
            ema_decay=0.9,
            save_every_steps=1,
        ),
    )
