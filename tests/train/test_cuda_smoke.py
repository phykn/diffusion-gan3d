import math

import pytest
import torch

from src.build import build_models, build_optimizers
from src.diffusion import Diffusion
from src.train.config import (
    AnchorConfig,
    DataConfig,
    DiffusionConfig,
    LoopConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from src.train.ema import build_ema
from src.train.engine import Trainer


class _CudaStream:
    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    def next(self) -> torch.Tensor:
        return self.labels.clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_64_cube_training_step_fits_six_gibibytes() -> None:
    device = torch.device("cuda")
    data = DataConfig(
        folder={0: ".", 1: ".", 2: "."},
        crop_size=64,
        patch_size=64,
        num_phases=3,
        batch_size=8,
    )
    model = ModelConfig(
        base_channels=16,
        channel_multipliers=(1, 2, 4, 4),
        embedding_channels=128,
        latent_channels=64,
        critic_channels=(32, 64, 128, 256),
        gradient_checkpointing=True,
    )
    optim = OptimConfig(
        generator_lr=0.00016,
        critic_lr=0.0001,
        beta1=0.5,
        beta2=0.9,
        r1_gamma=0.05,
        r1_interval=16,
        critic_local_weight=0.5,
    )
    cfg = TrainConfig(
        data=data,
        model=model,
        diffusion=DiffusionConfig(
            timesteps=11,
            beta_min=0.1,
            beta_max=20.0,
        ),
        anchor=AnchorConfig(probability=1.0, loss_weight=1.0),
        optim=optim,
        train=LoopConfig(
            steps=1,
            volume_batch_size=1,
            slices_per_axis=8,
            mixed_precision=True,
            ema_decay=0.999,
            save_every_steps=1,
        ),
    )
    denoiser, critics = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg,
    )
    labels = torch.randint(
        0,
        data.num_phases,
        (data.batch_size, 64, 64),
    )
    trainer = Trainer(
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams={axis: _CudaStream(labels) for axis in (0, 1, 2)},
        diffusion=Diffusion(
            11,
            beta_min=0.1,
            beta_max=20.0,
        ).to(device),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=True),
        device=device,
        volume_batch_size=cfg.train.volume_batch_size,
        num_phases=cfg.data.num_phases,
        patch_size=cfg.data.patch_size,
        slices_per_axis=cfg.train.slices_per_axis,
        ema_decay=cfg.train.ema_decay,
        r1_gamma=cfg.optim.r1_gamma,
        r1_interval=cfg.optim.r1_interval,
        critic_local_weight=cfg.optim.critic_local_weight,
        anchor_probability=cfg.anchor.probability,
        anchor_loss_weight=cfg.anchor.loss_weight,
        anchor_max_planes=cfg.anchor.max_planes,
        latent_channels=cfg.model.latent_channels,
        amp_enabled=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    metrics = trainer.step(15, transition=0)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    assert math.isfinite(metrics.generator)
    assert math.isfinite(metrics.critic)
    assert metrics.anchor_planes >= 1
    assert math.isfinite(metrics.anchor_loss)
    assert metrics.r1 > 0.0
    assert peak < 6 * 1024**3
    print(f"peak CUDA allocation: {peak / 1024**3:.2f} GiB")
