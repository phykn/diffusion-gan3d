import math

import pytest
import torch

from src.build.model import build_models
from src.build.trainer import build_optimizers
from src.model.diffusion import Diffusion
from src.prepare.resize import phase_channels
from src.train.burn_in import BurnIn
from src.train.ema import build_ema
from src.train.loss.denoiser import DenoiserLossSettings
from src.train.trainer import Trainer, TrainerComponents, TrainerSettings


class CudaStream:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def next(self) -> dict:
        return {"image": self.images.clone()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_64_cube_training_step_fits_six_gibibytes() -> None:
    device = torch.device("cuda")
    cfg = {
        "data": {
            "domains": {0: {"xy": ["."], "xz": ["."], "yz": ["."]}},
            "crop_size": 64,
            "num_phases": 3,
            "lo_res_size": 64,
        },
        "model": {
            "generator": {
                "channels": (16, 32, 64, 64),
                "latent_channels": 64,
                "embedding_channels": 128,
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
                                0: {"xy": ["."], "xz": ["."], "yz": ["."]}
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
            "cfg_dropout_probability": 0.1,
        },
        "loss": {
            "volume_fraction_weight": 1.0,
            "r1_weight": 0.05,
            "r1_every_steps": 16,
        },
    }
    data = cfg["data"]
    model = cfg["model"]
    optim = cfg["optim"]
    train = cfg["train"]
    denoiser, critics = build_models(cfg)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(denoiser, critics, cfg)
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
            streams={
                0: {
                    axis: CudaStream(phase_channels(images, data["num_phases"]))
                    for axis in (0, 1, 2)
                }
            },
            diffusion=Diffusion(11, beta_min=0.1, beta_max=20.0).to(device),
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            scaler=torch.amp.GradScaler("cuda", enabled=True),
            device=device,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            num_phases=data["num_phases"],
            patch_size=data["lo_res_size"],
            slice_pairs_per_plane=train["slice_pairs_per_plane"],
            # A constant schedule: the configured values from the first step.
            burn_in=BurnIn(
                steps=0,
                generator_lr=optim["generator_lr"],
                critic_lr=optim["critic_lr"],
                betas=tuple(optim["adam_betas"]),
                ema_decay=optim["ema_decay"],
                lr_scale=1.0,
                beta2=optim["adam_betas"][1],
                regularization_scale=1.0,
            ),
            r1_weight=cfg["loss"]["r1_weight"],
            r1_every_steps=cfg["loss"]["r1_every_steps"],
            loss=DenoiserLossSettings(
                vf_weight=cfg["loss"]["volume_fraction_weight"],
                real_transition_weight=0.0,
                profile_bins=16,
                profile_weight=0.0,
                profile_gradient_weight=0.0,
                consistency_weight=0.0,
                consistency_tolerance=0.0,
                num_phases=data["num_phases"],
                group_statistics_weight=0.0,
                group_statistics_max_gap=8,
                group_statistics_tolerance=0.01,
                preserve_height=False,
                mirror_balance_weight=0.0,
                mirror_balance_max_gap=4,
                mirror_balance_tolerance=0.02,
            ),
            domain_keep_probability=cfg["conditioning"]["domain_keep_probability"],
            cfg_dropout_probability=cfg["conditioning"]["cfg_dropout_probability"],
            latent_channels=model["generator"]["latent_channels"],
            amp_enabled=True,
            rollout_amp=True,
            r2_weight=0.0,
            group_statistics_start_step=0,
            group_statistics_ramp_steps=5000,
            real_transition_max_gap=1,
            real_transition_start_step=0,
            real_transition_ramp_steps=20000,
            structure_every_steps=100,
            coarse_corruption_probability=0.0,
            coarse_corruption_strength=0.0,
            height_data=None,
            profile_settings={
                "enabled": False,
                "num_bins": 16,
                "critic_enabled": False,
            },
            volume_permute_z=True,
            volume_flip_z=True,
        ),
    )

    torch.cuda.reset_peak_memory_stats(device)
    metrics = trainer.step(0)

    assert math.isfinite(metrics.generator_total)
    assert torch.cuda.max_memory_allocated(device) <= 6 * 1024**3
