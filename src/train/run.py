"""Run folders and the training loop: saves, checkpoints, interrupts, resume."""

import signal
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Event, current_thread, main_thread

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.io import PROJECT_ROOT, project_path, save_yaml, stored_path
from src.config.schema import StageConfig, update
from src.console import field, paint, path_field, section, warning
from src.data.images import fingerprint, load_images
from src.export import check_lr_source, export_config, read_export, write_export
from src.train.checkpoint import (
    check_fingerprint,
    load_checkpoint,
    relocate,
    resolve_checkpoint,
    restore,
    save_checkpoint,
    saved_config,
)
from src.train.metrics import selection_metrics, write_metrics

DONE = ".done"


def stage_dir(name: Path, stage: str) -> Path:
    return name.with_name(f"{name.name}-{stage}")


def new_run_name(nickname: str, root: Path | None = None) -> Path:
    """run/<YYYYMMDD-HHMM>-<nickname>, with _NN when that name is taken."""
    root = PROJECT_ROOT / "run" if root is None else root
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    base = f"{stamp}-{nickname}" if nickname else stamp
    sequence = 1
    while True:
        name = root / (base if sequence == 1 else f"{base}_{sequence:02d}")
        if not any(stage_dir(name, stage).exists() for stage in ("lr", "sr")):
            return name
        sequence += 1


def is_done(run_dir: Path) -> bool:
    return (run_dir / DONE).is_file()


def mark_done(run_dir: Path) -> None:
    (run_dir / DONE).write_text("", encoding="utf-8")


def training_images(config: StageConfig) -> dict:
    folders = {
        plane: [project_path(path) for path in paths]
        for plane, paths in config.data.images.items()
    }
    return load_images(folders, config.data.num_phases)


def build_trainer(
    config: StageConfig, device: torch.device, planes=None, *, lr=None, payload=None
):
    """The stage trainer with its selection.

    SR needs its frozen LR export: lr for a new run, which also builds the
    bank, or payload, a checkpoint whose LR export and bank it resumes.
    """
    from src.train.lr import LRTrainer
    from src.train.selection import (
        CheckpointSelection,
        SRCheckpointSelection,
        build_reference,
    )
    from src.train.sr import Bank, SRTrainer

    planes = training_images(config) if planes is None else planes
    if config.stage == "lr":
        trainer = LRTrainer(config, device, planes)
    elif payload is not None:
        trainer = SRTrainer(
            config, device, planes, payload["lr"], Bank.from_state(payload["bank"])
        )
    else:
        trainer = SRTrainer(config, device, planes, lr)
    if config.selection.enabled:
        reference = build_reference(trainer.samplers)
        samples = config.selection.samples
        if config.stage == "lr":
            trainer.selection = CheckpointSelection(
                reference, samples, config.conditioning.height_enabled
            )
        else:
            # A resume restores the inputs its earlier steps were scored on.
            trainer.selection = SRCheckpointSelection.from_bank(
                reference, samples, trainer.bank
            )
    return trainer


def current_payload(trainer) -> dict:
    return trainer.export_payload(trainer.completed_steps, trainer.ema.state_dict())


def save_export(trainer, path: Path, *, overwrite=True) -> Path:
    return write_export(path, current_payload(trainer), overwrite=overwrite)


def save_weights(trainer, root: Path) -> None:
    """generator.pt: the best scored step with selection, else the latest."""
    selection = trainer.selection
    if selection is not None and selection.best() is not None:
        write_export(root / "generator.pt", selection.best_payload(trainer))
    else:
        save_export(trainer, root / "generator.pt")


def select_step(trainer, root: Path, writer) -> None:
    entry = trainer.selection.update(trainer, root)
    best = trainer.selection.best()
    write_metrics(writer, entry["step"], selection_metrics(entry, best))
    line = f"step {entry['step']:>7}  structure {entry['structure']:.3f}"
    if best["step"] == entry["step"]:
        tqdm.write(paint(f"{line}  * new best", "green"))
    else:
        best_text = f"best {best['structure']:.3f} @ {best['step']}"
        tqdm.write(f"{line}  {paint(best_text, 'dim')}")


@contextmanager
def deferred_sigint():
    """Let a training step finish before handling a terminal interrupt.

    A second interrupt stops at once; run_train then saves nothing mid-step.
    """
    requested = Event()

    def interrupt(*_):
        if requested.is_set():
            raise KeyboardInterrupt
        requested.set()
        warning("Stopping after the current step; press Ctrl+C again to stop now.")

    in_main_thread = current_thread() is main_thread()
    if in_main_thread:
        previous = signal.signal(signal.SIGINT, interrupt)
    try:
        yield requested
    finally:
        if in_main_thread:
            signal.signal(signal.SIGINT, previous)


def run_train(trainer, run_dir: Path) -> Path:
    """Train from trainer.completed_steps to train.total_steps in run_dir.

    Every weights_every_steps: score (with selection) and write generator.pt.
    Every archive_every_steps and at the end: a checkpoint. The first SIGINT
    finishes the step, saves both and stops; a second stops at once.
    """
    config = trainer.config
    total = config.train.total_steps
    start = trainer.completed_steps
    if not 0 <= start < total:
        raise ValueError("train.total_steps must exceed the completed steps.")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "train.yaml", config)
    if trainer.selection is not None:
        trainer.selection.restore(trainer, run_dir)
    every = config.train.weights_every_steps
    archive = config.train.archive_every_steps
    with (
        deferred_sigint() as stop,
        SummaryWriter(run_dir / "tensorboard") as writer,
        tqdm(
            range(start, total),
            desc=f"{config.stage.upper()} train",
            dynamic_ncols=True,
        ) as bar,
    ):
        at_boundary = True
        checkpoint_step = None
        try:
            for step in bar:
                if stop.is_set():
                    raise KeyboardInterrupt
                at_boundary = False
                metrics = trainer.step(step)
                at_boundary = True
                done = step + 1
                write_metrics(writer, done, metrics)
                writer.flush()
                # An interrupt while saving leaves that save incomplete; saving
                # again would collide with its files, so stop without it.
                at_boundary = False
                save_now = done % every == 0 or done == total
                # Score before the checkpoint so a resume keeps this score.
                if save_now and trainer.selection is not None:
                    select_step(trainer, run_dir, writer)
                    writer.flush()
                if done == total or done % archive == 0:
                    save_checkpoint(trainer, run_dir, current_payload(trainer))
                    checkpoint_step = done
                if save_now:
                    save_weights(trainer, run_dir)
                at_boundary = True
                if stop.is_set():
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            # An exception inside a step may leave partly updated optimizers;
            # never save that state.
            if at_boundary:
                if checkpoint_step != trainer.completed_steps:
                    save_checkpoint(trainer, run_dir, current_payload(trainer))
                save_weights(trainer, run_dir)
            raise
    mark_done(run_dir)
    return run_dir


def describe(config: StageConfig, run_dir: Path) -> None:
    section(f"{config.stage.upper()} training")
    data = config.data
    field(
        "Resolution",
        f"crop {data.crop_size} px -> LR {data.lo_res_size} voxels, "
        f"HR {data.hi_res_size} voxels",
    )
    if config.stage == "sr":
        source = config.source
        field("LR source", f"{source.lr_run} (step {source.lr_step})")
    path_field("Run", run_dir)


def sr_config(config: StageConfig, lr_dir: Path) -> tuple[StageConfig, dict]:
    """The SR config pointing at the finished LR run, and that run's export."""
    lr = read_export(lr_dir, "lr")
    check_lr_source(export_config(lr), config)
    config = update(config, "source.lr_run", stored_path(lr_dir))
    return update(config, "source.lr_step", lr["step"]), lr


def train_stage(config: StageConfig, device, run_dir: Path, lr_dir=None) -> Path:
    """Train one new stage into run_dir; SR trains from the LR run lr_dir."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    lr = None
    if config.stage == "sr":
        config, lr = sr_config(config, lr_dir)
    trainer = build_trainer(config, device, lr=lr)
    describe(config, run_dir)
    return run_train(trainer, run_dir)


def read_resume(resume: Path, path_map=None) -> tuple[dict, StageConfig]:
    """A completed checkpoint, relocated, and its config under this schema."""
    payload = relocate(load_checkpoint(resolve_checkpoint(resume)), path_map)
    return payload, saved_config(payload)


def resume_stage(
    resume: Path, device, run_dir: Path, *, steps=None, path_map=None
) -> tuple[str, Path]:
    """Continue one stage from a checkpoint into the new run_dir."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    payload, config = read_resume(resume, path_map)
    if steps is not None:
        config = update(config, "train.total_steps", steps)
    if config.train.total_steps <= payload["step"]:
        raise ValueError(f"--steps must exceed the {payload['step']} completed steps.")
    planes = training_images(config)
    check_fingerprint(payload["data_fingerprint"], fingerprint(planes))
    trainer = build_trainer(config, device, planes, payload=payload)
    restore(trainer, payload)
    describe(config, run_dir)
    path_field("Resume", resume)
    return config.stage, run_train(trainer, run_dir)
