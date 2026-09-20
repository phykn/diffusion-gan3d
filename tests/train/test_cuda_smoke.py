import math

import pytest
import torch

from src.build.model import build_models
from src.build.trainer import build_optimizers
from src.model.diffusion import Diffusion
from src.prepare.resize import phase_channels
from src.train.ema import build_ema
from src.train.trainer import Trainer, TrainerComponents, TrainerSettings


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
            "domains": {0: {"xy": ".", "xz": ".", "yz": "."}},
            "crop_size": 64,
            "num_phases": 3,
            "lo_res_size": 64,
        },
        "model": {
            "generator": {
                "channels": (16, 32, 64, 64),
                "latent_channels": 64,
                "embedding_channels": 128,
                "anchor_multiscale_input": False,
            },
            "critic": {
                "channels": (32, 64, 128, 256),
                "plane_groups": [
                    [plane]
                    for plane in ("xy", "xz", "yz")
                    if any(
                        (
                            plane in planes
                            for planes in {
                                0: {"xy": ".", "xz": ".", "yz": "."}
                            }.values()
                        )
                    )
                ],
            },
            "gradient_checkpointing": True,
            "diffusion": {"num_steps": 11, "beta_min": 0.1, "beta_max": 20.0},
        },
        "optim": {
            "generator_lr": 0.00016,
            "critic_lr": 0.0001,
            "adam_betas": (0.5, 0.9),
            "ema_decay": 0.999,
        },
        "train": {
            "volume_batch_size": 1,
            "real_batch_size": 8,
            "total_steps": 1,
            "mixed_precision": True,
            "slice_pairs_per_plane": 8,
            "weights_every_steps": 1,
        },
        "conditioning": {
            "domain_keep_probability": 1.0,
            "anchor": {
                "probability": 1.0,
                "start_step": 0,
                "ramp_steps": 0,
                "borrowed_plane_probability": 0.0,
            },
        },
        "loss": {
            "anchor_pixel_weight": 0.05,
            "connectivity": {
                "adversarial_weight": 0.0,
                "normal_transition_weight": 0.0,
            },
            "volume_fraction_weight": 1.0,
            "critic_local_weight": 0.5,
            "r1_weight": 0.05,
            "r1_every_steps": 16,
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
        (train["real_batch_size"], 64, 64),
    )
    trainer = Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema,
            critics=critics,
            connectivity_critic=connectivity_critic,
            streams={
                0: {
                    axis: CudaStream(phase_channels(images, data["num_phases"]))
                    for axis in (0, 1, 2)
                }
            },
            diffusion=Diffusion(11, beta_min=0.1, beta_max=20.0).to(device),
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            connectivity_optim=connectivity_optim,
            scaler=torch.amp.GradScaler("cuda", enabled=True),
            device=device,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            num_phases=data["num_phases"],
            patch_size=data["lo_res_size"],
            slice_pairs_per_axis=train["slice_pairs_per_plane"],
            ema_decay=optim["ema_decay"],
            r1_gamma=cfg["loss"]["r1_weight"],
            r1_interval=cfg["loss"]["r1_every_steps"],
            critic_local_weight=cfg["loss"]["critic_local_weight"],
            anchor_training_probability=cfg["conditioning"]["anchor"]["probability"],
            anchor_start_step=cfg["conditioning"]["anchor"]["start_step"],
            anchor_ramp_steps=cfg["conditioning"]["anchor"]["ramp_steps"],
            anchor_pixel_loss_weight=cfg["loss"]["anchor_pixel_weight"],
            anchor_shared_axis_probability=cfg["conditioning"]["anchor"][
                "borrowed_plane_probability"
            ],
            connectivity_weight=cfg["loss"]["connectivity"]["adversarial_weight"],
            normal_transition_weight=cfg["loss"]["connectivity"][
                "normal_transition_weight"
            ],
            vf_loss_weight=cfg["loss"]["volume_fraction_weight"],
            domain_dropout=1.0 - cfg["conditioning"]["domain_keep_probability"],
            cfg_drop_each_probability=0.0,
            latent_channels=model["generator"]["latent_channels"],
            amp_enabled=True,
        ),
    )

    torch.cuda.reset_peak_memory_stats(device)
    metrics = trainer.step(0)

    assert math.isfinite(metrics.generator_total)
    assert torch.cuda.max_memory_allocated(device) <= 6 * 1024**3
