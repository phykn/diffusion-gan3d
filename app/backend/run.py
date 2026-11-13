import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.src.app import create_app
from app.backend.src.config import DEFAULT_CONFIG, load_server_config


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the 3D inference API.")
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="SR generator.pt or its run directory",
    )
    parser.add_argument(
        "--lr-weight",
        type=Path,
        help="LR generator.pt or run directory (default: the LR stored in the SR export)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        help="inference device; defaults to CUDA or MPS when available",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_server_config(args.config)
    app = create_app(
        args.weight, device=args.device, lr_weights=args.lr_weight, config=config
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
