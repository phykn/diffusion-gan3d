import sys
from datetime import datetime

from src.config.build import change_setting
from src.config.io import PROJECT_ROOT, dump_yaml, resolve_path, shorten_path
from src.config.sample import build_stage_configs, load_sample
from src.console import field, path_field, section, success
from src.data.images import load_images
from src.device import check_device
from src.export import check_lr_source, export_config, read_export
from src.train.checkpoint import (
    load_checkpoint,
    relocate,
    resolve_checkpoint,
    saved_config,
)
from src.train.loop import DONE, TrainingRun
from src.train.lr import LRTrainer
from src.train.sr import Bank, SRTrainer


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


def stage_configs(args):
    configs = build_stage_configs(load_sample(args.sample or "default"))
    if args.steps is not None:
        configs["lr"] = change_setting(configs["lr"], "train.total_steps", args.steps)
    if args.sr_steps is not None:
        configs["sr"] = change_setting(
            configs["sr"], "train.total_steps", args.sr_steps
        )
    if args.bank_size is not None:
        configs["sr"] = change_setting(configs["sr"], "lr_bank.size", args.bank_size)
    return configs


def show(configs, run_name):
    sr = configs["sr"]
    if run_name is not None:
        lr_run = str(stage_dir(run_name, "lr"))
        source = sr.source.model_copy(update={"lr_run": lr_run})
        sr = sr.model_copy(update={"source": source})
    for stage, config in (("lr", configs["lr"]), ("sr", sr)):
        print(f"# {stage}\n{dump_yaml(config)}")


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


def stage_dir(name, stage):
    return name.with_name(f"{name.name}-{stage}")


def is_done(run_dir):
    return (run_dir / DONE).is_file()


def train_stage(config, device, run_dir, lr_dir=None):
    """Train one new stage into run_dir; SR trains from the LR run lr_dir."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    lr = None
    if config.stage == "sr":
        config, lr = sr_config(config, lr_dir)
    trainer = build_trainer(config, device, lr=lr)
    training = TrainingRun(trainer, run_dir)
    describe(config, run_dir)
    return training.train()


def sr_config(config, lr_dir):
    """The SR config pointing at the finished LR run, and that run's export."""
    lr = read_export(lr_dir, "lr")
    check_lr_source(export_config(lr), config)
    config = change_setting(config, "source.lr_run", shorten_path(lr_dir))
    return change_setting(config, "source.lr_step", lr["step"]), lr


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


def resume_stage(resume, device, run_dir, steps=None, path_map=None):
    """Continue one stage from a checkpoint into the new run_dir."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    payload = relocate(load_checkpoint(resolve_checkpoint(resume)), path_map)
    config = saved_config(payload)
    if steps is not None:
        config = change_setting(config, "train.total_steps", steps)
    if config.train.total_steps <= payload["step"]:
        raise ValueError(f"--steps must exceed the {payload['step']} completed steps.")
    trainer = build_trainer(config, device, payload=payload)
    training = TrainingRun(trainer, run_dir, payload)
    describe(config, run_dir)
    path_field("Resume", resume)
    return config.stage, training.train()


def build_trainer(config, device, lr=None, payload=None):
    """The stage's trainer; SR takes its LR export, or both from a checkpoint."""
    planes = training_images(config)
    if config.stage == "lr":
        return LRTrainer(config, device, planes)
    if payload is not None:
        return SRTrainer(
            config, device, planes, payload["lr"], Bank.from_state(payload["bank"])
        )
    return SRTrainer(config, device, planes, lr)


def training_images(config):
    folders = {
        plane: [resolve_path(path) for path in paths]
        for plane, paths in config.data.images.items()
    }
    return load_images(folders, config.data.num_phases)


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
