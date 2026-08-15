from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .. import AXES
from .engine import Metrics, Trainer
from .weights import GENERATOR_FILE, save_all_weights, save_checkpoint


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
    weights = root / GENERATOR_FILE
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
                weights = save_all_weights(
                    root,
                    trainer.ema_denoiser,
                    trainer.critics,
                    trainer.connectivity_critic,
                )
            if checkpoint_every is not None and done % checkpoint_every == 0:
                checkpoint = save_checkpoint(
                    root,
                    done,
                    trainer.ema_denoiser,
                    trainer.critics,
                    trainer.connectivity_critic,
                )
                print(f"Saved checkpoint: {checkpoint}")
        if done % save_every:
            weights = save_all_weights(
                root,
                trainer.ema_denoiser,
                trainer.critics,
                trainer.connectivity_critic,
            )
    except KeyboardInterrupt:
        weights = save_all_weights(
            root,
            trainer.ema_denoiser,
            trainer.critics,
            trainer.connectivity_critic,
        )
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
        "loss/relation": metrics.relation_loss,
        "loss/relation_raw": metrics.relation_loss,
        "loss/relation_weighted": metrics.relation_weighted_loss,
        "loss/relation_phase": metrics.relation_phase_loss,
        "loss/relation_support": metrics.relation_support_loss,
        "loss/relation_minus": metrics.relation_minus_loss,
        "loss/relation_plus": metrics.relation_plus_loss,
        "train/transition": metrics.transition,
        "train/volume_size": metrics.volume_size,
        "train/domain": metrics.domain,
        "train/connectivity_triplets": metrics.connectivity_triplets,
        "train/connectivity_replay": metrics.connectivity_replay,
        "train/teacher_volumes": metrics.teacher_volumes,
        "train/teacher_mebibytes": metrics.teacher_mebibytes,
        "train/relation_queries": metrics.relation_queries,
        "train/relation_matches": metrics.relation_matches,
        "train/relation_domain_matches": metrics.relation_domain_matches,
        "train/relation_shared_matches": metrics.relation_shared_matches,
        "train/relation_ood_rejections": metrics.relation_ood_rejections,
        "train/relation_missing_references": (
            metrics.relation_missing_references
        ),
        "train/relation_match_rate": (
            metrics.relation_matches / max(metrics.relation_queries, 1)
        ),
        "train/relation_ood_reject_rate": (
            metrics.relation_ood_rejections / max(metrics.relation_queries, 1)
        ),
        "train/relation_time_weight": metrics.relation_time_weight,
        "train/relation_bank_entries": metrics.relation_bank_entries,
        "train/relation_ready_buckets": metrics.relation_ready_buckets,
        "train/relation_prior_ready": float(metrics.relation_prior_ready),
        "conditioning/anchor_planes": metrics.anchor_planes,
        "conditioning/anchor_ramp": metrics.anchor_ramp,
        "conditioning/anchor_input_active_fraction": (
            metrics.anchor_input_active_fraction
        ),
        "conditioning/anchor_teacher": float(metrics.anchor_teacher),
        "conditioning/anchor_shared": float(metrics.anchor_shared),
        "conditioning/vf_active": float(metrics.vf_active),
        "conditioning/vf_active_fraction": metrics.vf_active_fraction,
        "conditioning/vf_target_resample_rate": metrics.vf_target_resample_rate,
        "conditioning/vf_hard_mae": metrics.hard_vf_mae,
    }
    for tag, value in scalars.items():
        writer.add_scalar(tag, value, step)

    for distance in (1, 2, 4):
        if distance <= len(metrics.relation_distance_weights):
            writer.add_scalar(
                f"train/relation_distance_weight_d{distance}",
                metrics.relation_distance_weights[distance - 1],
                step,
            )
    if metrics.relation_distance_weights:
        weighted_distance = sum(
            distance * weight
            for distance, weight in enumerate(
                metrics.relation_distance_weights,
                start=1,
            )
        )
        writer.add_scalar(
            "train/relation_weighted_distance",
            weighted_distance,
            step,
        )

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
    for phase, values in enumerate(
        zip(
            metrics.target_vfs,
            metrics.target_vf_stds,
            metrics.soft_vfs,
            metrics.hard_vfs,
            strict=True,
        )
    ):
        target, target_std, soft, hard = values
        writer.add_scalar(f"conditioning/vf_target_{phase}", target, step)
        writer.add_scalar(f"conditioning/vf_target_std_{phase}", target_std, step)
        writer.add_scalar(f"conditioning/vf_soft_{phase}", soft, step)
        writer.add_scalar(f"conditioning/vf_hard_{phase}", hard, step)
