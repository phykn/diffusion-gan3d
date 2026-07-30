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
    load_checkpoint,
    load_train_config,
    save_checkpoint,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train.yaml"


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

    if cfg.train.checkpoint is None:
        run_dir = _new_run_dir(Path(cfg.output.run_root))
        start_step = 0
        save_mapping(run_dir / "train.yaml", cfg.as_dict())
    else:
        run_dir = Path(cfg.train.checkpoint).parent
        start_step = load_checkpoint(
            cfg.train.checkpoint,
            denoiser=denoiser,
            ema_denoiser=ema_denoiser,
            critics=_critic_sequence(critics),
            denoiser_optimizer=denoiser_optimizer,
            critic_optimizers=critic_optimizers,
            scaler=scaler,
            config_signature=cfg.resume_signature(),
        )
    if start_step >= cfg.train.steps:
        print(
            f"Checkpoint already contains {start_step} completed steps; "
            f"train.steps={cfg.train.steps}."
        )
        return

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
    )

    completed = start_step
    print(
        f"Training Diffusion GAN3D steps={start_step}->{cfg.train.steps} "
        f"device={device} run={run_dir}"
    )
    writer = SummaryWriter(run_dir / "tensorboard")
    progress = tqdm(
        range(start_step, cfg.train.steps),
        initial=start_step,
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
            )
            if completed % cfg.train.save_every_steps == 0:
                _save(
                    run_dir,
                    completed,
                    cfg,
                    denoiser,
                    ema_denoiser,
                    critics,
                    denoiser_optimizer,
                    critic_optimizers,
                    scaler,
                )
        _save(
            run_dir,
            completed,
            cfg,
            denoiser,
            ema_denoiser,
            critics,
            denoiser_optimizer,
            critic_optimizers,
            scaler,
        )
    except KeyboardInterrupt:
        path = _save(
            run_dir,
            completed,
            cfg,
            denoiser,
            ema_denoiser,
            critics,
            denoiser_optimizer,
            critic_optimizers,
            scaler,
        )
        print(f"Training interrupted after step {completed}; checkpoint={path}")
    finally:
        progress.close()
        writer.close()


def _new_run_dir(root: Path) -> Path:
    name = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _critic_sequence(critics: torch.nn.ModuleDict) -> tuple[torch.nn.Module, ...]:
    return tuple(critics[str(axis)] for axis in AXES)


def _save(
    run_dir,
    step,
    cfg,
    denoiser,
    ema_denoiser,
    critics,
    denoiser_optimizer,
    critic_optimizers,
    scaler,
) -> Path:
    return save_checkpoint(
        run_dir,
        step=step,
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=_critic_sequence(critics),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=scaler,
        config_signature=cfg.resume_signature(),
    )


def _write_metrics(writer, step, metrics) -> None:
    writer.add_scalar("loss/generator", metrics.generator, step)
    writer.add_scalar("loss/critic_total", metrics.critic, step)
    writer.add_scalar("loss/r1_raw", metrics.r1, step)
    writer.add_scalar("train/transition", metrics.transition, step)
    for axis, value in zip(AXES, metrics.critic_axes, strict=True):
        writer.add_scalar(f"loss/critic_axis_{axis}", value, step)


if __name__ == "__main__":
    main()
