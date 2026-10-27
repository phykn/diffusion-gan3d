import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simul.src.export import generate
from src.config.files import load_yaml
from src.console import field, path_field, section

DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Simulation YAML (default: simul/config.yaml).",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    export = generate(cfg)
    slice_count = sum(len(paths) for paths in export.slices.values())
    section("Simulated data")
    field("Volumes", len(export.volumes))
    field("Slices", slice_count)
    for key in export.measured[0]:
        mean = sum(item[key] for item in export.measured) / len(export.measured)
        field(key, f"{mean:.3f} measured, {cfg['geometry'][key]} target")
    path_field("Output", cfg["output"]["data_dir"])


if __name__ == "__main__":
    main()
