import argparse
from pathlib import Path

from src.device import default_device
from src.train.run.low_res import run_low_res_train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LR model; defaults come from config/train/low_res.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Training YAML; defaults to config/train/low_res.yaml.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint file or run/checkpoints directory (latest completed step).",
    )
    parser.add_argument(
        "--path-map",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Relocate saved path prefixes on resume; repeat for multiple roots.",
    )
    parser.add_argument(
        "--steps", type=int, help="Total target steps, including completed steps."
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Data YAML relative to the project root; overrides config.data.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=default_device(),
        help="Compute device; CUDA, then Apple MPS, then CPU when available.",
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
