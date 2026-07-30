import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.build import build_axis_streams, build_models, build_optimizers
from src.data import AXES
from src.diffusion import DiffusionProcess
from src.misc import save_mapping
from src.train import (
    DiffusionGANTrainer,
    build_ema,
    load_train_config,
    save_model_weights,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train.yaml"
RUN_ROOT = Path(__file__).resolve().parent / "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    denoiser, critics = build_models(cfg.data, cfg.model)
    denoiser = denoiser.to(device)
    critics = critics.to(device)
    ema_denoiser = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg.optim,
    )
    amp_enabled = cfg.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    diffusion = DiffusionProcess(
        cfg.diffusion.timesteps,
        beta_min=cfg.diffusion.beta_min,
        beta_max=cfg.diffusion.beta_max,
    ).to(device)

    run_dir = _new_run_dir(RUN_ROOT)
    save_mapping(run_dir / "train.yaml", cfg.as_dict())

    streams = build_axis_streams(cfg.data, device=device)
    trainer = DiffusionGANTrainer(
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=critics,
        streams=streams,
        diffusion=diffusion,
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=scaler,
        num_phases=cfg.data.num_phases,
        patch_size=cfg.data.patch_size,
        latent_channels=cfg.model.latent_channels,
        volume_batch_size=cfg.train.volume_batch_size,
        slices_per_axis=cfg.train.slices_per_axis,
        mixed_precision=cfg.train.mixed_precision,
        ema_decay=cfg.train.ema_decay,
        r1_gamma=cfg.optim.r1_gamma,
        r1_interval=cfg.optim.r1_interval,
        device=device,
        anchor_probability=cfg.anchor.probability,
        anchor_weight=cfg.anchor.loss_weight,
    )

    completed = 0
    print(
        f"Training Diffusion GAN3D steps=0->{cfg.train.steps} "
        f"device={device} run={run_dir}"
    )
    writer = SummaryWriter(run_dir / "tensorboard")
    progress = tqdm(
        range(cfg.train.steps),
        total=cfg.train.steps,
        desc="Diffusion GAN3D",
        dynamic_ncols=True,
    )
    try:
        for step in progress:
            metrics = trainer.train_step(step)
            completed = step + 1
            _write_metrics(writer, completed, metrics)
            progress.set_postfix(
                G=f"{metrics.generator:.4g}",
                D=f"{metrics.critic:.4g}",
                t=metrics.transition,
                A=int(metrics.anchor_used),
            )
            if completed % cfg.train.save_every_steps == 0:
                _save(run_dir, ema_denoiser, critics)
        _save(run_dir, ema_denoiser, critics)
    except KeyboardInterrupt:
        path = _save(run_dir, ema_denoiser, critics)
        print(f"Training interrupted after step {completed}; weights={path}")
    finally:
        progress.close()
        writer.close()


def _new_run_dir(root: Path) -> Path:
    name = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save(
    run_dir: Path,
    ema_denoiser: torch.nn.Module,
    critics: torch.nn.ModuleDict,
) -> Path:
    return save_model_weights(run_dir, ema_denoiser, critics)


def _write_metrics(writer, step, metrics) -> None:
    writer.add_scalar("loss/generator", metrics.generator, step)
    writer.add_scalar("loss/generator_total", metrics.generator_total, step)
    writer.add_scalar("loss/critic_total", metrics.critic, step)
    writer.add_scalar("loss/r1_raw", metrics.r1, step)
    writer.add_scalar("train/transition", metrics.transition, step)
    writer.add_scalar("conditioning/anchor_used", metrics.anchor_used, step)
    if metrics.anchor_used:
        writer.add_scalar("loss/anchor", metrics.anchor_loss, step)
        writer.add_scalar(
            "conditioning/anchor_accuracy",
            metrics.anchor_accuracy,
            step,
        )
    for axis, value in zip(AXES, metrics.critic_axes, strict=True):
        writer.add_scalar(f"loss/critic_axis_{axis}", value, step)


if __name__ == "__main__":
    main()
