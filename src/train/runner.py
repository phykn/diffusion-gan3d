from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .. import AXES
from ..utils import save_model
from .engine import Metrics, Trainer


def save_models(root: Path, trainer: Trainer) -> Path:
    generator = save_model(root / "generator.pt", trainer.ema_denoiser)
    for axis, critic in trainer.critics.items():
        save_model(root / f"critic_{axis}.pt", critic)
    save_model(root / "critic_c.pt", trainer.connectivity_critic)
    return generator


def run_training(
    trainer: Trainer,
    *,
    steps: int,
    save_every: int,
    run_dir: str | Path,
    checkpoint_every: int | None = None,
) -> Path:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if (
        not isinstance(save_every, int)
        or isinstance(save_every, bool)
        or save_every < 1
    ):
        raise ValueError("save_every must be a positive integer.")
    if checkpoint_every is not None and (
        not isinstance(checkpoint_every, int)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every < 1
    ):
        raise ValueError("checkpoint_every must be a positive integer or None.")

    root = Path(run_dir)
    done = 0
    weights = root / "generator.pt"
    print("\nTraining")
    print("--------")
    print(f"Steps  : {steps}")
    print(f"Device : {trainer.device}")
    print(f"Run    : {root}")
    writer = SummaryWriter(root / "tensorboard")
    bar = tqdm(
        range(steps),
        total=steps,
        desc="Diffusion GAN3D",
        dynamic_ncols=True,
    )
    try:
        for step in bar:
            metrics = trainer.step(step)
            done = step + 1
            write_metrics(writer, done, metrics)
            bar.set_postfix(
                G=f"{metrics.generator:.4g}",
                D=f"{metrics.critic:.4g}",
                t=metrics.transition,
                S=metrics.volume_size,
                A=metrics.anchor_planes,
            )
            if done % save_every == 0:
                weights = save_models(root, trainer)
            if checkpoint_every is not None and done % checkpoint_every == 0:
                checkpoint = save_models(
                    root / "checkpoints" / f"step_{done:08d}",
                    trainer,
                )
                print(f"Saved checkpoint: {checkpoint}")
        if done % save_every:
            weights = save_models(root, trainer)
    except KeyboardInterrupt:
        weights = save_models(root, trainer)
        print(f"Training interrupted after step {done}; weights={weights}")
        raise
    finally:
        bar.close()
        writer.close()
    return weights


def write_metrics(writer: SummaryWriter, step: int, metrics: Metrics) -> None:
    scalars = {
        "loss/generator": metrics.generator,
        "loss/generator_total": metrics.generator_total,
        "loss/generator_global": metrics.generator_global,
        "loss/generator_local_raw": metrics.generator_local,
        "loss/critic_total": metrics.critic,
        "loss/critic_global": metrics.critic_global,
        "loss/critic_local_raw": metrics.critic_local,
        "loss/generator_connectivity": metrics.generator_connectivity,
        "loss/critic_connectivity": metrics.critic_connectivity,
        "loss/connectivity_r1_raw": metrics.connectivity_r1,
        "loss/r1_raw": metrics.r1,
        "loss/vf": metrics.vf_loss,
        "loss/normal_transition": metrics.normal_transition_loss,
        "loss/anchor_coarse": metrics.anchor_coarse_loss,
        "loss/anchor_pixel": metrics.anchor_pixel_loss,
        "train/transition": metrics.transition,
        "train/volume_size": metrics.volume_size,
        "train/domain": metrics.domain,
        "train/connectivity_triplets": metrics.connectivity_triplets,
        "conditioning/anchor_planes": metrics.anchor_planes,
        "conditioning/anchor_ramp": metrics.anchor_ramp,
        "conditioning/anchor_input_active_fraction": (
            metrics.anchor_input_active_fraction
        ),
        "conditioning/anchor_shared": float(metrics.anchor_shared),
        "conditioning/vf_active": float(metrics.vf_active),
        "conditioning/vf_active_fraction": metrics.vf_active_fraction,
    }
    for tag, value in scalars.items():
        writer.add_scalar(tag, value, step)

    states = ("both", "anchor_only", "vf_only", "joint_null")
    for name, fraction in zip(
        states,
        metrics.condition_state_fractions,
        strict=True,
    ):
        writer.add_scalar(f"conditioning/state_{name}_fraction", fraction, step)

    if metrics.anchor_planes:
        anchor_scalars = {
            "loss/anchor": metrics.anchor_loss,
            "conditioning/anchor_accuracy": metrics.anchor_accuracy,
            "conditioning/anchor_conflict_rate": metrics.anchor_conflict_rate,
            f"loss/anchor_{metrics.anchor_planes}_planes": metrics.anchor_loss,
            f"conditioning/anchor_accuracy_{metrics.anchor_planes}_planes": (
                metrics.anchor_accuracy
            ),
        }
        for tag, value in anchor_scalars.items():
            writer.add_scalar(tag, value, step)

    for axis, value in zip(AXES, metrics.critic_axes, strict=True):
        writer.add_scalar(f"loss/critic_axis_{axis}", value, step)
