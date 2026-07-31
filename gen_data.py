import argparse
from pathlib import Path

from src.build import generate_data
from src.simul.config import SimulationConfig, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "simul.yaml"


def parse_args(argv: list[str] | None = None) -> SimulationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        return load_config(args.config)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def main() -> None:
    cfg = parse_args()
    result = generate_data(cfg)
    slice_count = sum(len(paths) for paths in result.slices.values())
    print(f"volumes={len(result.volumes)}")
    print(f"slices={slice_count} dir={cfg.output.data_dir}")


if __name__ == "__main__":
    main()
