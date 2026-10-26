import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import add_domain_seed_arguments, result_directory
from scripts.common.diagnostic import show_labels
from src.build.data import build_datasets
from src.build.predict import resolve_weights
from src.config.files import find_saved_config, load_yaml
from src.config.sample import build_train_config, find_sample, load_sample
from src.config.train import load_train_config
from src.plane import PLANES

SAMPLES = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect training crops of a sample or of a trained run."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--weight",
        type=Path,
        help="Use the data saved with this run's weights.",
    )
    source.add_argument(
        "--sample",
        default="default",
        help="Name in config/samples/ or a sample YAML path (default: default).",
    )
    parser.add_argument(
        "--stage",
        choices=("low_res", "sr"),
        default="low_res",
        help="With --sample: sr shows the HR crops (default: low_res).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Preview PNG (default: a new directory under checks/).",
    )
    add_domain_seed_arguments(parser)
    parser.add_argument(
        "--no-view", action="store_true", help="Save results without a viewer."
    )
    args = parser.parse_args()
    if args.no_view:
        plt.switch_backend("Agg")
    if args.weight:
        config = find_saved_config(resolve_weights(args.weight))
        stage = load_yaml(config)["stage"]
        cfg = load_train_config(config, stage)
    else:
        config, stage = find_sample(args.sample), args.stage
        cfg = load_train_config(build_train_config(load_sample(config), stage), stage)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = build_datasets(cfg, high_resolution=stage == "sr")
    if args.domain not in datasets:
        parser.error(f"--domain must be one of {list(datasets)}")
    print(f"Config  : {config.resolve()}")
    axes = tuple(datasets[args.domain])
    fig, panels = plt.subplots(
        len(axes),
        SAMPLES,
        squeeze=False,
        figsize=(3 * SAMPLES, 2.5 * len(axes)),
    )
    for row, axis in enumerate(axes):
        ds = datasets[args.domain][axis]
        for col in range(SAMPLES):
            group = ds.path_groups[np.random.randint(len(ds.path_groups))]
            path = group[np.random.randint(len(group))]
            img = ds[path]["image"].argmax(0)
            show_labels(panels[row, col], img, cfg["data"]["num_phases"], PLANES[axis])
    fig.suptitle("Random training crops (dominant phase; training retains fractions)")
    fig.tight_layout()
    output = args.out or result_directory(__file__) / "crops.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Preview : {output.resolve()}")
    if not args.no_view:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
