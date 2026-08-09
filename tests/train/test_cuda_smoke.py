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
            "folders": {0: ".", 1: ".", 2: "."},
            "crop_size": 64,
            "input_size": 64,
            "num_phases": 3,
            "batch_size": 8,
        },
        "model": {
            "base_channels": 16,
            "channel_multipliers": (1, 2, 4, 4),
            "embedding_channels": 128,
            "latent_channels": 64,
            "critic_channels": (32, 64, 128, 256),
            "gradient_checkpointing": True,
        },
        "diffusion": {"timesteps": 11, "beta_min": 0.1, "beta_max": 20.0},
        "anchor": {
            "training_probability": 1.0,
            "start_step": 0,
            "ramp_steps": 0,
            "multi_anchor_prob": 0.5,
            "max_density": 0.05,
            "min_spacing": 4,
            "mixed_axis_prob": 0.5,
            "teacher_bank_size_mib": 16,
            "loss_weight": 1.0,
        },
        "connectivity": {
            "loss_weight": 0.0,
            "replay_triplets_per_axis": 1,
            "replay_capacity_per_axis": 2,
            "max_triplets_per_step": 1,
            "reversal_invariant": True,
            "normal_transition_loss_weight": 0.0,
        },
        "vf": {"loss_weight": 1.0},
        "optim": {
            "denoiser_lr": 0.00016,
            "critic_lr": 0.0001,
            "beta1": 0.5,
            "beta2": 0.9,
            "r1_gamma": 0.05,
            "r1_interval": 16,
            "local_loss_weight": 0.5,
        },
        "train": {
            "total_steps": 1,
            "volume_batch_size": 1,
            "volume_sizes": (64,),
            "slice_pairs_per_axis": 8,
            "mixed_precision": True,
            "ema_decay": 0.999,
            "save_every_steps": 1,
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
        data["num_phases"],
        (data["batch_size"], 64, 64),
    )
    trainer = Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema,
            critics=critics,
            connectivity_critic=connectivity_critic,
            streams={axis: CudaStream(images) for axis in (0, 1, 2)},
            diffusion=Diffusion(11, beta_min=0.1, beta_max=20.0).to(device),
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            connectivity_optim=connectivity_optim,
            scaler=torch.amp.GradScaler("cuda", enabled=True),
            device=device,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            volume_sizes=train["volume_sizes"],
            num_phases=data["num_phases"],
            patch_size=data["input_size"],
            slice_pairs_per_axis=train["slice_pairs_per_axis"],
            ema_decay=train["ema_decay"],
            r1_gamma=optim["r1_gamma"],
            r1_interval=optim["r1_interval"],
            critic_local_weight=optim["local_loss_weight"],
            anchor_training_probability=cfg["anchor"]["training_probability"],
            anchor_start_step=cfg["anchor"]["start_step"],
            anchor_ramp_steps=cfg["anchor"]["ramp_steps"],
            anchor_multi_probability=cfg["anchor"]["multi_anchor_prob"],
            anchor_max_density=cfg["anchor"]["max_density"],
            anchor_min_spacing=cfg["anchor"]["min_spacing"],
            anchor_mixed_axis_probability=cfg["anchor"]["mixed_axis_prob"],
            anchor_teacher_bank_mebibytes=cfg["anchor"]["teacher_bank_size_mib"],
            anchor_loss_weight=cfg["anchor"]["loss_weight"],
            connectivity_weight=cfg["connectivity"]["loss_weight"],
            normal_transition_weight=cfg["connectivity"][
                "normal_transition_loss_weight"
            ],
            connectivity_replay_triplets_per_axis=cfg["connectivity"][
                "replay_triplets_per_axis"
            ],
            connectivity_replay_capacity_per_axis=cfg["connectivity"][
                "replay_capacity_per_axis"
            ],
            connectivity_max_triplets_per_step=cfg["connectivity"][
                "max_triplets_per_step"
            ],
            vf_loss_weight=cfg["vf"]["loss_weight"],
            cfg_drop_each_probability=0.0,
            cfg_single_drop_probability=0.1,
            scale_consistency_overlap=4,
            scale_consistency_probability=0.0,
            scale_consistency_start_step=0,
            scale_consistency_ramp_steps=0,
            scale_consistency_weight=0.0,
            latent_channels=model["latent_channels"],
            amp_enabled=True,
        ),
    )

    torch.cuda.reset_peak_memory_stats(device)
    metrics = trainer.step(0)

    assert math.isfinite(metrics.generator_total)
    assert torch.cuda.max_memory_allocated(device) <= 6 * 1024**3
