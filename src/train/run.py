import signal
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from threading import Event, current_thread, main_thread

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.io import PROJECT_ROOT, dump_yaml, project_path, save_yaml, stored_path
from src.config.sample import build_stage_configs, load_sample
from src.config.schema import to_dict, update
from src.console import field, paint, path_field, section, success, warning
from src.data.images import fingerprint, load_images
from src.device import check_device
from src.export import check_lr_source, export_config, read_export, write_export
from src.train.checkpoint import (
    FORMAT,
    VERSION,
    check_fingerprint,
    load_checkpoint,
    relocate,
    resolve_checkpoint,
    save_checkpoint,
    saved_config,
)
from src.train.lr import LRTrainer
from src.train.metrics import selection_metrics, write_metrics
from src.train.selection import (
    CheckpointSelection,
    SRCheckpointSelection,
    build_reference,
)
from src.train.sr import Bank, SRTrainer

DONE = ".done"


def stage_dir(name, stage):
    return name.with_name(f"{name.name}-{stage}")


def new_run_name(nickname):
    """run/<YYYYMMDD-HHMM>-<nickname>, with _NN when that name is taken."""
    root = PROJECT_ROOT / "run"
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    base = f"{stamp}-{nickname}" if nickname else stamp
    sequence = 1
    while True:
        name = root / (base if sequence == 1 else f"{base}_{sequence:02d}")
        if not any(stage_dir(name, stage).exists() for stage in ("lr", "sr")):
            return name
        sequence += 1


def is_done(run_dir):
    return (run_dir / DONE).is_file()


def training_images(config):
    folders = {
        plane: [project_path(path) for path in paths]
        for plane, paths in config.data.images.items()
    }
    return load_images(folders, config.data.num_phases)


def build_trainer(config, device, planes=None, *, lr=None, payload=None):
    planes = training_images(config) if planes is None else planes
    if config.stage == "lr":
        return LRTrainer(config, device, planes)
    if payload is not None:
        return SRTrainer(
            config, device, planes, payload["lr"], Bank.from_state(payload["bank"])
        )
    return SRTrainer(config, device, planes, lr)


class TrainingRun:
    def __init__(self, trainer, root, data_fingerprint, payload=None):
        self.trainer = trainer
        self.root = root
        self.data_fingerprint = data_fingerprint
        self.path_maps = [] if payload is None else payload["path_maps"]
        config = trainer.config
        reference = build_reference(trainer.samplers)
        samples = config.selection.samples
        if config.stage == "lr":
            self.selection = CheckpointSelection(
                reference, samples, config.conditioning.height_enabled
            )
        else:
            self.selection = SRCheckpointSelection.from_bank(
                reference, samples, trainer.bank
            )
        if payload is not None:
            trainer.load_state_dict(payload)
            self.selection.load_state_dict(payload["selection"])

    def current_export(self):
        trainer = self.trainer
        return trainer.export_payload(trainer.completed_steps, trainer.ema.state_dict())

    def save_weights(self):
        payload = (
            self.selection.best_payload(self.trainer)
            if self.selection.best()
            else self.current_export()
        )
        write_export(self.root / "generator.pt", payload)

    def archive(self):
        trainer = self.trainer
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "stage": trainer.config.stage,
            "step": trainer.completed_steps,
            "config": to_dict(trainer.config),
            "selection": self.selection.state_dict(),
            "data_fingerprint": self.data_fingerprint,
            "path_maps": self.path_maps,
            **trainer.state_dict(),
        }
        save_checkpoint(payload, self.root, self.current_export())

    def score(self, writer):
        entry = self.selection.update(self.trainer, self.root)
        best = self.selection.best()
        write_metrics(writer, entry["step"], selection_metrics(entry, best))
        line = f"step {entry['step']:>7}  structure {entry['structure']:.3f}"
        if best["step"] == entry["step"]:
            tqdm.write(paint(f"{line}  * new best", "green"))
        else:
            best_text = f"best {best['structure']:.3f} @ {best['step']}"
            tqdm.write(f"{line}  {paint(best_text, 'dim')}")

    def train(self):
        trainer = self.trainer
        run_dir = self.root
        config = trainer.config
        total = config.train.total_steps
        start = trainer.completed_steps
        if not 0 <= start < total:
            raise ValueError("train.total_steps must exceed the completed steps.")
        run_dir.mkdir(parents=True, exist_ok=False)
        save_yaml(run_dir / "train.yaml", config)
        self.selection.restore(trainer, run_dir)
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
                    # A save interrupted halfway must not be attempted twice.
                    at_boundary = False
                    save_now = done % every == 0 or done == total
                    if save_now:
                        self.score(writer)
                        writer.flush()
                    if done == total or done % archive == 0:
                        self.archive()
                        checkpoint_step = done
                    if save_now:
                        self.save_weights()
                    at_boundary = True
                    if stop.is_set():
                        raise KeyboardInterrupt
            except KeyboardInterrupt:
                if at_boundary:
                    if checkpoint_step != trainer.completed_steps:
                        self.archive()
                    self.save_weights()
                raise
        (run_dir / DONE).touch()
        return run_dir


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


def describe(config, run_dir):
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


def sr_config(config, lr_dir):
    """The SR config pointing at the finished LR run, and that run's export."""
    lr = read_export(lr_dir, "lr")
    check_lr_source(export_config(lr), config)
    config = update(config, "source.lr_run", stored_path(lr_dir))
    return update(config, "source.lr_step", lr["step"]), lr


def train_stage(config, device, run_dir, lr_dir=None):
    """Train one new stage into run_dir; SR trains from the LR run lr_dir."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    lr = None
    if config.stage == "sr":
        config, lr = sr_config(config, lr_dir)
    planes = training_images(config)
    trainer = build_trainer(config, device, planes, lr=lr)
    run = TrainingRun(trainer, run_dir, fingerprint(planes))
    describe(config, run_dir)
    return run.train()


def resume_stage(resume, device, run_dir, *, steps=None, path_map=None):
    """Continue one stage from a checkpoint into the new run_dir."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    payload = relocate(load_checkpoint(resolve_checkpoint(resume)), path_map)
    config = saved_config(payload)
    if steps is not None:
        config = update(config, "train.total_steps", steps)
    if config.train.total_steps <= payload["step"]:
        raise ValueError(f"--steps must exceed the {payload['step']} completed steps.")
    planes = training_images(config)
    data_fingerprint = fingerprint(planes)
    check_fingerprint(payload["data_fingerprint"], data_fingerprint)
    trainer = build_trainer(config, device, planes, payload=payload)
    run = TrainingRun(trainer, run_dir, data_fingerprint, payload)
    describe(config, run_dir)
    path_field("Resume", resume)
    return config.stage, run.train()


def stage_configs(args):
    configs = build_stage_configs(load_sample(args.sample or "default"))
    if args.steps is not None:
        configs["lr"] = update(configs["lr"], "train.total_steps", args.steps)
    if args.sr_steps is not None:
        configs["sr"] = update(configs["sr"], "train.total_steps", args.sr_steps)
    if args.bank_size is not None:
        configs["sr"] = update(configs["sr"], "lr_bank.size", args.bank_size)
    return configs


def show(configs, run_name):
    sr = configs["sr"]
    if run_name is not None:
        lr_run = str(stage_dir(run_name, "lr"))
        sr = replace(sr, source=replace(sr.source, lr_run=lr_run))
    for stage, config in (("lr", configs["lr"]), ("sr", sr)):
        print(f"# {stage}\n{dump_yaml(to_dict(config))}")


def stage_dirs(name):
    """NAME-lr and NAME-sr; an unfinished one stops with how to continue it."""
    dirs = stage_dir(name, "lr"), stage_dir(name, "sr")
    for path in dirs:
        if path.exists() and not is_done(path):
            try:
                resolve_checkpoint(path)
            except FileNotFoundError:
                sys.exit(f"{path} exists but saved no checkpoint; delete it: {path}")
            sys.exit(
                f"{path} exists but did not finish. Continue it with "
                f"python train.py --resume {path}/checkpoints --run-dir <new name>, "
                f"or delete it: {path}"
            )
    return dirs


def resume(args):
    device = check_device(args.device)
    stage = load_checkpoint(resolve_checkpoint(args.resume))["stage"]
    lr_dir, sr_dir = stage_dirs(args.run_dir)
    run_dir = lr_dir if stage == "lr" else sr_dir
    stage, run_dir = resume_stage(
        args.resume, device, run_dir, steps=args.steps, path_map=args.path_map
    )
    success(f"{stage.upper()} training finished: {run_dir}")
    if stage == "lr":
        path_field("Next", f"python train.py <sample> --run-dir {args.run_dir}  (SR)")


def run(args):
    if args.resume is not None:
        resume(args)
        return
    configs = stage_configs(args)
    if args.show:
        show(configs, args.run_dir)
        return
    device = check_device(args.device)
    # Both stages are checked before LR starts, so a bad SR setting leaves no run.
    check_lr_source(configs["lr"], configs["sr"])
    name = (
        args.run_dir
        if args.run_dir is not None
        else new_run_name(configs["lr"].nickname)
    )
    lr_dir, sr_dir = stage_dirs(name)
    if is_done(lr_dir):
        section("LR training")
        path_field("Reuse finished", lr_dir)
    else:
        train_stage(configs["lr"], device, lr_dir)
    if args.lr_only:
        success(f"LR training finished: {lr_dir}")
        return
    if is_done(sr_dir):
        section("SR training")
        path_field("Reuse finished", sr_dir)
    else:
        train_stage(configs["sr"], device, sr_dir, lr_dir=lr_dir)
    success(f"Training finished: {lr_dir}, {sr_dir}")
