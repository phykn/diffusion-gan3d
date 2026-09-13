import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml
from src.simul.export import generate

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "simul.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    export = generate(cfg)
    slice_count = sum(len(paths) for paths in export.slices.values())
    print(f"Volumes : {len(export.volumes)}")
    print(f"Slices  : {slice_count}")
    print(f"Output  : {cfg['output']['data_dir']}")


if __name__ == "__main__":
    main()
