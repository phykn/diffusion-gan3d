import argparse
from pathlib import Path

from src.simul.export import generate
from src.utils import load_yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "simul.yaml"


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
