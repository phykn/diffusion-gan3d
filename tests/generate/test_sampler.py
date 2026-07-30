from dataclasses import replace
from pathlib import Path

import torch

from src.build import build_models
from src.generate import (
    PlaneAnchor,
    generate_labels,
    latest_model_weights,
    load_denoiser_weights,
)
from src.misc import save_mapping
from src.train import build_ema, save_model_weights
from src.train.config import (
    AnchorConfig,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    TrainingConfig,
)


def test_ema_weights_generate_categorical_volume(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run" / "sample"
    run_dir.mkdir(parents=True)
    save_mapping(run_dir / "train.yaml", cfg.as_dict())
    denoiser, critics = build_models(cfg.data, cfg.model)
    ema = build_ema(denoiser)
    weights = save_model_weights(run_dir, ema, critics)

    loaded, loaded_cfg = load_denoiser_weights(
        weights,
        device=torch.device("cpu"),
    )
    labels = generate_labels(
        loaded,
        loaded_cfg,
        device=torch.device("cpu"),
    )

    assert latest_model_weights(tmp_path / "run") == weights
    assert labels.shape == (8, 8, 8)
    assert labels.dtype == torch.uint8
    assert int(labels.max()) < cfg.data.num_phases


def test_anchor_aware_weights_accept_soft_plane_condition(
    tmp_path: Path,
) -> None:
    cfg = replace(
        _config(tmp_path),
        anchor=AnchorConfig(probability=0.5, loss_weight=1.0),
    )
    run_dir = tmp_path / "run" / "anchored"
    run_dir.mkdir(parents=True)
    save_mapping(run_dir / "train.yaml", cfg.as_dict())
    denoiser, critics = build_models(cfg.data, cfg.model)
    weights = save_model_weights(
        run_dir,
        build_ema(denoiser),
        critics,
    )
    loaded, loaded_cfg = load_denoiser_weights(
        weights,
        device=torch.device("cpu"),
    )
    anchor = PlaneAnchor(
        labels=torch.randint(
            0,
            cfg.data.num_phases,
            (8, 8),
        ),
        axis=1,
        index=4,
    )

    labels = generate_labels(
        loaded,
        loaded_cfg,
        device=torch.device("cpu"),
        anchors=(anchor,),
    )

    assert labels.shape == (8, 8, 8)


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
            timesteps=2,
            beta_min=0.1,
            beta_max=2.0,
        ),
        anchor=AnchorConfig(
            probability=0.0,
            loss_weight=0.0,
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
            steps=10,
            volume_batch_size=1,
            slices_per_axis=2,
            mixed_precision=False,
            ema_decay=0.9,
            save_every_steps=1,
        ),
    )
