import argparse
from pathlib import Path

import torch

from src.train.run.sr import run_sr_train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a separate 3D super-resolution model from stage-1 volumes and real 2D slices."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, help="Data YAML for a new run.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--base-weights", type=Path, help="Override source.weights in the SR config."
    )
    source.add_argument("--resume", type=Path)
    parser.add_argument(
        "--path-map",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Relocate saved path prefixes on resume; repeat for multiple roots.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Directory for this run (created as new; default: run/MMDDHHMM_sr[_nickname]).",
    )
    parser.add_argument("--steps", type=int, help="Total step target, also on resume.")
    parser.add_argument(
        "--bank-size",
        type=int,
        help="Number of frozen LR samples per domain on a new run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_sr_train(**vars(parse_args(argv)))


if __name__ == "__main__":
    main()
