import math

import pytest
import torch

from src.build import build_models, build_optimizers
from src.diffusion import DiffusionProcess
from src.train import DiffusionGANTrainer, build_ema
from src.train.config import DataConfig, ModelConfig, OptimConfig


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
    )
    denoiser, critics = build_models(data, model)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        optim,
    )
    labels = torch.randint(
        0,
        data.num_phases,
        (data.batch_size, 64, 64),
    )
    trainer = DiffusionGANTrainer(
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams={axis: _CudaStream(labels) for axis in (0, 1, 2)},
        diffusion=DiffusionProcess(
            11,
            beta_min=0.1,
            beta_max=20.0,
        ).to(device),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=True),
        num_phases=data.num_phases,
        patch_size=data.patch_size,
        latent_channels=model.latent_channels,
        volume_batch_size=1,
        slices_per_axis=8,
        mixed_precision=True,
        ema_decay=0.999,
        r1_gamma=optim.r1_gamma,
        r1_interval=16,
        device=device,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    metrics = trainer.train_step(15, transition=0)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    assert math.isfinite(metrics.generator)
    assert math.isfinite(metrics.critic)
    assert metrics.r1 > 0.0
    assert peak < 6 * 1024**3
    print(f"peak CUDA allocation: {peak / 1024**3:.2f} GiB")
