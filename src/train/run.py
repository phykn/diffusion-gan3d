import json
from dataclasses import asdict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.plane import PLANES
from src.storage import save_model
from src.train.state import save_training
from src.train.trainer import Metrics, Trainer


def run_train(
    trainer: Trainer,
    steps: int,
    save_every: int,
    run_dir: str | Path,
    checkpoint_every: int | None = None,
    start_step: int = 0,
) -> Path:
    for name, value in (
        ("steps", steps),
        ("save_every", save_every),
        ("checkpoint_every", checkpoint_every),
    ):
        if name == "checkpoint_every" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    root = Path(run_dir)
    if not 0 <= start_step < steps:
        raise ValueError("start_step must be less than total steps.")
    done = start_step
    weights = root / "generator.pt"
    writer = SummaryWriter(root / "tensorboard")
    bar = tqdm(
        range(start_step, steps),
        desc="Diffusion GAN3D",
        dynamic_ncols=True,
    )
    model_files = {
        "generator.pt": trainer.ema_denoiser,
        **{f"critic_{plane}.pt": critic for plane, critic in trainer.critics.items()},
        "critic_c.pt": trainer.connectivity_critic,
    }
    log = (root / "metrics.jsonl").open("w", encoding="utf-8")
    try:
        for step in bar:
            metrics = trainer.step(step)
            done = step + 1
            write_metrics(writer, done, metrics)
            log.write(
                json.dumps({"step": done, **asdict(metrics)}, allow_nan=False) + "\n"
            )
            log.flush()
            if done % save_every == 0:
                for name, model in model_files.items():
                    save_model(root / name, model)
                if hasattr(trainer, "cfg"):
                    save_training(root / "checkpoints" / "last.pt", trainer)
            if checkpoint_every is not None and done % checkpoint_every == 0:
                checkpoint_root = root / "checkpoints" / f"step_{done:08d}"
                for name, model in model_files.items():
                    save_model(checkpoint_root / name, model)
        if done % save_every:
            for name, model in model_files.items():
                save_model(root / name, model)
            if hasattr(trainer, "cfg"):
                save_training(root / "checkpoints" / "last.pt", trainer)
    except KeyboardInterrupt:
        for name, model in model_files.items():
            save_model(root / name, model)
        raise
    finally:
        log.close()
        bar.close()
        writer.close()
    return weights


def write_metrics(writer: SummaryWriter, step: int, metrics: Metrics) -> None:
    scalars = {
        "loss/generator": metrics.generator,
        "loss/generator_total": metrics.generator_total,
        "loss/critic": metrics.critic,
        "loss/r1": metrics.r1,
        "loss/generator_connectivity": metrics.generator_connectivity,
        "loss/critic_connectivity": metrics.critic_connectivity,
        "loss/connectivity_r1": metrics.connectivity_r1,
        "loss/normal_transition": metrics.normal_transition_loss,
        "loss/anchor": metrics.anchor_loss,
        "loss/vf": metrics.vf_loss,
        "conditioning/anchor_fraction": metrics.anchor_input_active_fraction,
        "conditioning/vf_fraction": metrics.vf_active_fraction,
        "conditioning/anchor_ramp": metrics.anchor_ramp,
        "sampling/transition": metrics.transition,
        f"timestep/{metrics.transition}/generator": metrics.generator,
        f"timestep/{metrics.transition}/critic": metrics.critic,
        **{
            f"critic_plane/{PLANES[axis]}": value
            for axis, value in enumerate(metrics.critic_axes)
        },
        **metrics.diagnostics,
    }

    if metrics.anchor_planes:
        scalars["conditioning/anchor_planes"] = metrics.anchor_planes
        scalars["conditioning/anchor_accuracy"] = metrics.anchor_accuracy

    for tag, value in scalars.items():
        if value is not None:
            writer.add_scalar(tag, value, step)
