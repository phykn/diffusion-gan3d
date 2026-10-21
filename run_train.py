import argparse
import sys
from pathlib import Path

from src.config.files import dump_yaml, find_saved_config, load_yaml
from src.config.sample import STAGE_NAMES, build_train_config, load_sample
from src.device import default_device
from src.train.checkpoint import resolve_checkpoint
from src.train.run.low_res import run_low_res_train
from src.train.run.sr import run_sr_train

DONE = ".done"


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
        help="Print the LR and SR training settings built from the sample and exit.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Run name: trains NAME-lr then NAME-sr and reuses a finished stage "
            "(default: new run/MMDDHHMM_<stage>_<sample> directories)."
        ),
    )
    parser.add_argument("--steps", type=int, help="LR total steps.")
    parser.add_argument("--sr-steps", type=int, help="SR total steps.")
    parser.add_argument(
        "--bank-size", type=int, help="Frozen LR samples per domain for SR."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Continue one interrupted stage from a checkpoint file, a run "
            "directory, or its checkpoints/; --steps sets its total steps."
        ),
    )
    parser.add_argument(
        "--path-map",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Relocate saved path prefixes on resume; repeat for multiple roots.",
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
        args.resume = args.resume.expanduser()
    elif args.path_map:
        parser.error("--path-map requires --resume.")
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser()
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resume is not None:
        resume(args)
        return
    sample = load_sample(args.sample or "default")
    configs = {stage: build_train_config(sample, stage) for stage in STAGE_NAMES}
    if args.show:
        for stage, cfg in configs.items():
            print(f"# {stage}\n{dump_yaml(cfg)}")
        return
    lr_dir, sr_dir = get_stage_dirs(args.run_dir)
    if is_done(lr_dir):
        print(f"LR: reuse finished {lr_dir}", flush=True)
    else:
        lr_dir = run_low_res_train(
            args.device, config=configs["low_res"], run_dir=lr_dir, steps=args.steps
        )
        mark_done(lr_dir)
    if is_done(sr_dir):
        print(f"SR: {sr_dir} already finished", flush=True)
        return
    sr_dir = run_sr_train(
        args.device,
        base_weights=lr_dir,
        config=configs["sr"],
        run_dir=sr_dir,
        steps=args.sr_steps,
        bank_size=args.bank_size,
    )
    mark_done(sr_dir)
    print(f"Done: LR {lr_dir}, SR {sr_dir}", flush=True)


def resume(args: argparse.Namespace) -> None:
    stage = load_yaml(find_saved_config(resolve_checkpoint(args.resume)))["stage"]
    lr_dir, sr_dir = get_stage_dirs(args.run_dir)
    options = dict(resume=args.resume, steps=args.steps, path_map=args.path_map)
    if stage == "low_res":
        run_dir = run_low_res_train(args.device, run_dir=lr_dir, **options)
        mark_done(run_dir)
        if args.run_dir is not None:
            print(
                f"LR finished. Next: python run_train.py <sample> --run-dir "
                f"{args.run_dir} trains its SR model.",
                flush=True,
            )
    else:
        run_dir = run_sr_train(args.device, run_dir=sr_dir, **options)
        mark_done(run_dir)


def get_stage_dirs(run_dir: Path | None) -> tuple[Path | None, Path | None]:
    if run_dir is None:
        return None, None
    lr_dir, sr_dir = (Path(f"{run_dir}-{name}") for name in STAGE_NAMES.values())
    for path in (lr_dir, sr_dir):
        if path.exists() and not is_done(path):
            sys.exit(
                f"{path} exists but did not finish. Continue it with --resume "
                f"{path}/checkpoints --run-dir <new name>, or delete it."
            )
    return lr_dir, sr_dir


def is_done(path: Path | None) -> bool:
    return path is not None and (path / DONE).is_file()


def mark_done(path: Path) -> None:
    (path / DONE).touch()


if __name__ == "__main__":
    main()
