import argparse
from pathlib import Path

import uvicorn

from src.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3D inference API.")
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="generator.pt or its run directory",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="inference device; defaults to CUDA when available",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.weight, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
