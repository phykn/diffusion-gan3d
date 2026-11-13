import argparse
import sys
from pathlib import Path

from src.device import default_device
from src.train.run import run


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
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


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
