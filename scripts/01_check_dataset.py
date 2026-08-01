"""Show random crops from each training axis."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import AXES
from src.build import build_datasets
from src.train.config import load_config

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive.")

    cfg = load_config(args.config)
    datasets = build_datasets(cfg)
    fig, panels = plt.subplots(
        len(AXES),
        args.samples,
        squeeze=False,
        figsize=(3 * args.samples, 8),
    )
    for row, axis in enumerate(AXES):
        ds = datasets[axis]
        for col in range(args.samples):
            img = ds[np.random.randint(len(ds))].numpy()
            panels[row, col].imshow(
                img,
                cmap="gray",
                vmin=-0.5,
                vmax=cfg.data.num_phases - 0.5,
                interpolation="nearest",
            )
            panels[row, col].set_title(f"axis {axis}")
            panels[row, col].axis("off")
    fig.suptitle("Random training crops")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
