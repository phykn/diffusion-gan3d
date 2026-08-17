import math

import pytest
import torch

from src.build import build_models, build_optimizers
from src.diffusion import Diffusion
from src.train.ema import build_ema
from src.train.engine import Trainer, TrainerComponents, TrainerSettings


class CudaStream:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def next(self) -> torch.Tensor:
        return self.images.clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_64_cube_training_step_fits_six_gibibytes() -> None:
    device = torch.device("cuda")
    cfg = {
        "data": {
            "domains": {0: {0: ".", 1: ".", 2: "."}},
            "num_phase": 3,
            "crop_partial": False,
            "crop_size": 64,
            "input_size": 64,
            "domain_prob": 1.0,
            "batch_size": 8,
        },
        "model": {
            "grad_checkpoint": True,
            "generator": {
                "channels": (16, 32, 64, 64),
                "condition_channels": 128,
                "latent_channels": 64,
            },
            "critic": {
                "channels": (32, 64, 128, 256),
                "local_loss_weight": 0.5,
                "r1_weight": 0.05,
                "r1_interval": 16,
            },
        },
        "diffusion": {"steps": 11, "beta_min": 0.1, "beta_max": 20.0},
        "anchor": {
            "multiscale_input": False,
            "train_prob": 1.0,
            "start_step": 0,
            "ramp_steps": 0,
            "cross_domain_prob": 0.0,
            "pixel_weight": 0.05,
            "connectivity": {
                "weight": 0.0,
                "volume_count": 1,
                "refresh_every": 500,
                "phase_transition_weight": 0.0,
            },
        },
        "vf": {"max_samples": 4, "weight": 1.0},
        "optim": {
            "generator_lr": 0.00016,
            "critic_lr": 0.0001,
            "adam_betas": (0.5, 0.9),
            "ema_decay": 0.999,
        },
        "train": {
            "steps": 1,
            "volume_batch_size": 1,
            "pairs_per_axis": 8,
            "amp": True,
            "update_weights_every": 1,
        },
    }
    data = cfg["data"]
    model = cfg["model"]
    optim = cfg["optim"]
    train = cfg["train"]
    denoiser, critics, connectivity_critic = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    connectivity_critic = connectivity_critic.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    images = torch.randint(
        0,
        data["num_phase"],
        (data["batch_size"], 64, 64),
    )
    trainer = Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema,
            critics=critics,
            connectivity_critic=connectivity_critic,
            streams={0: {axis: CudaStream(images) for axis in (0, 1, 2)}},
            diffusion=Diffusion(11, beta_min=0.1, beta_max=20.0).to(device),
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            connectivity_optim=connectivity_optim,
            scaler=torch.amp.GradScaler("cuda", enabled=True),
            device=device,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            num_phases=data["num_phase"],
            patch_size=data["input_size"],
            slice_pairs_per_axis=train["pairs_per_axis"],
            ema_decay=optim["ema_decay"],
            r1_gamma=model["critic"]["r1_weight"],
            r1_interval=model["critic"]["r1_interval"],
            critic_local_weight=model["critic"]["local_loss_weight"],
            anchor_training_probability=cfg["anchor"]["train_prob"],
            anchor_start_step=cfg["anchor"]["start_step"],
            anchor_ramp_steps=cfg["anchor"]["ramp_steps"],
            anchor_pixel_loss_weight=cfg["anchor"]["pixel_weight"],
            anchor_shared_axis_probability=cfg["anchor"]["cross_domain_prob"],
            connectivity_weight=cfg["anchor"]["connectivity"]["weight"],
            normal_transition_weight=cfg["anchor"]["connectivity"][
                "phase_transition_weight"
            ],
            connectivity_bank_size=cfg["anchor"]["connectivity"]["volume_count"],
            connectivity_refresh_steps=cfg["anchor"]["connectivity"]["refresh_every"],
            vf_loss_weight=cfg["vf"]["weight"],
            vf_target_average_max_samples=cfg["vf"]["max_samples"],
            domain_dropout=1.0 - data["domain_prob"],
            cfg_drop_each_probability=0.0,
            latent_channels=model["generator"]["latent_channels"],
            amp_enabled=True,
        ),
    )

    torch.cuda.reset_peak_memory_stats(device)
    metrics = trainer.step(0)

    assert math.isfinite(metrics.generator_total)
    assert torch.cuda.max_memory_allocated(device) <= 6 * 1024**3
