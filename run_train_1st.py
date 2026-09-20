import argparse
from pathlib import Path

import torch

from src.train.run.low_res import run_low_res_train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LR model; defaults come from config/train/low_res.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Total target steps, including completed steps."
    )
    parser.add_argument("--data", type=Path, help="Data YAML; overrides config.data.")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "directory for this run (created as a new directory; "
            "default: run/MMDDHHMM_low_res[_nickname])"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_low_res_train(**vars(parse_args(argv)))


if __name__ == "__main__":
    main()
