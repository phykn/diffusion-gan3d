from pathlib import Path

import torch

from src.build import build_models, build_optimizers
from src.generate import generate_labels, latest_checkpoint, load_ema_denoiser
from src.misc import save_mapping
from src.train import build_ema, save_checkpoint
from src.train.config import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    OptimConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
)


def test_checkpoint_ema_generates_reproducible_categorical_volume(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run" / "sample"
    run_dir.mkdir(parents=True)
    save_mapping(run_dir / "train.yaml", cfg.as_dict())
    denoiser, critics = build_models(cfg.data, cfg.model)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg.optim,
    )
    checkpoint = save_checkpoint(
        run_dir,
        step=7,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=tuple(critics[str(axis)] for axis in (0, 1, 2)),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        config_signature=cfg.resume_signature(),
    )

    loaded, loaded_cfg, step = load_ema_denoiser(
        checkpoint,
        device=torch.device("cpu"),
    )
    first, seed = generate_labels(
        loaded,
        loaded_cfg,
        device=torch.device("cpu"),
        seed=123,
    )
    second, _ = generate_labels(
        loaded,
        loaded_cfg,
        device=torch.device("cpu"),
        seed=123,
    )

    assert latest_checkpoint(tmp_path / "run") == checkpoint
    assert step == 7
    assert seed == 123
    assert first.shape == (8, 8, 8)
    assert first.dtype == torch.uint8
    assert int(first.max()) < cfg.data.num_phases
    assert torch.equal(first, second)


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
        optim=OptimConfig(
            generator_lr=1e-3,
            critic_lr=1e-3,
            beta1=0.0,
            beta2=0.9,
            r1_gamma=0.0,
            r1_interval=2,
        ),
        train=TrainingConfig(
            checkpoint=None,
            steps=10,
            volume_batch_size=1,
            slices_per_axis=2,
            mixed_precision=False,
            ema_decay=0.9,
            save_every_steps=1,
        ),
        output=OutputConfig(run_root=root / "run"),
    )
