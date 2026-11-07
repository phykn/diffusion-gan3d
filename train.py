import argparse
import sys
from dataclasses import replace
from pathlib import Path

from src.config.io import dump_yaml
from src.config.sample import build_stage_configs, load_sample
from src.config.schema import to_dict, update
from src.console import path_field, section, success
from src.device import check_device, default_device
from src.export import check_lr_source
from src.train.checkpoint import load_checkpoint, resolve_checkpoint
from src.train.run import (
    is_done,
    new_run_name,
    resume_stage,
    stage_dir,
    train_stage,
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sample's LR model, then its SR model from it."
    )
    parser.add_argument(
        "sample",
        nargs="?",
        help="Name in config/samples/ or a sample YAML path (default: default).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the full LR and SR settings built from the sample and exit.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Run name such as run/NAME: trains NAME-lr then NAME-sr and reuses a "
            "finished stage (default: run/<YYYYMMDD-HHMM>-<sample>)."
        ),
    )
    parser.add_argument(
        "--lr-only",
        action="store_true",
        help="Stop after the LR stage; rerun without it and the same --run-dir for SR.",
    )
    parser.add_argument("--steps", type=positive_int, help="LR total steps.")
    parser.add_argument("--sr-steps", type=positive_int, help="SR total steps.")
    parser.add_argument(
        "--bank-size", type=positive_int, help="Frozen LR volumes the SR stage uses."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Continue one stage from a checkpoint file, a run folder or its "
            "checkpoints/ into a new --run-dir; --steps sets its total steps."
        ),
    )
    parser.add_argument(
        "--path-map",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Replace the saved path prefix OLD with NEW on resume (repeatable).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=default_device(),
        help="Compute device; CUDA, then Apple MPS, then CPU when available.",
    )
    args = parser.parse_args(argv)
    if args.resume is not None:
        if args.sample or args.show:
            parser.error("--resume uses the saved settings; omit the sample.")
        if args.sr_steps is not None or args.bank_size is not None:
            parser.error("--resume continues one stage; set its total with --steps.")
        if args.run_dir is None:
            parser.error("--resume needs --run-dir NAME for the continued run.")
        args.resume = args.resume.expanduser()
    elif args.path_map:
        parser.error("--path-map requires --resume.")
    if args.lr_only and (
        args.resume is not None
        or args.sr_steps is not None
        or args.bank_size is not None
    ):
        parser.error("--lr-only trains a new LR stage; omit --resume and SR options.")
    if args.lr_only and args.run_dir is None:
        parser.error("--lr-only needs --run-dir NAME; SR then continues from NAME-lr.")
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser()
    return args


def stage_configs(args: argparse.Namespace) -> dict:
    configs = build_stage_configs(load_sample(args.sample or "default"))
    if args.steps is not None:
        configs["lr"] = update(configs["lr"], "train.total_steps", args.steps)
    if args.sr_steps is not None:
        configs["sr"] = update(configs["sr"], "train.total_steps", args.sr_steps)
    if args.bank_size is not None:
        configs["sr"] = update(
            configs["sr"], "lr_bank.samples_per_domain", args.bank_size
        )
    return configs


def show(configs: dict, run_name: Path | None) -> None:
    sr = configs["sr"]
    if run_name is not None:
        lr_run = str(stage_dir(run_name, "lr"))
        sr = replace(sr, source=replace(sr.source, lr_run=lr_run))
    for stage, config in (("lr", configs["lr"]), ("sr", sr)):
        print(f"# {stage}\n{dump_yaml(to_dict(config))}")


def stage_dirs(name: Path) -> tuple[Path, Path]:
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


def resume(args: argparse.Namespace) -> None:
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
