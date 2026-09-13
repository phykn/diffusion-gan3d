import argparse
from pathlib import Path

import torch

from src.train.sr_run import run_sr_train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a separate 3D super-resolution model from stage-1 volumes and real 2D slices."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, help="Data YAML for a new run.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-weights", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=int, help="Total step target, also on resume.")
    parser.add_argument(
        "--bank-size",
        type=int,
        help="Number of frozen LR samples per domain on a new run.",
    )
    args = parser.parse_args(argv)
    run_sr_train(**vars(args))


if __name__ == "__main__":
    main()
