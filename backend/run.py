import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.app import create_app
from backend.src.config import DEFAULT_CONFIG, ServerConfig, load_server_config


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
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="inference device; defaults to CUDA or MPS when available",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-inflight-downloads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_server_config(args.config)
    if args.max_inflight_downloads is not None:
        config = ServerConfig.model_validate(
            {
                **config.model_dump(),
                "max_inflight_downloads": args.max_inflight_downloads,
            }
        )
    app = create_app(args.weight, device=args.device, config=config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
