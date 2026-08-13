import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build import build_datasets
from src.utils import load_yaml

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"
SAMPLES = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--domain",
        type=int,
        default=0,
        help="numeric domain ID (default: 0)",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    datasets = build_datasets(cfg)
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
            img = ds[np.random.randint(len(ds))].numpy()
            panels[row, col].imshow(
                img,
                cmap="gray",
                vmin=-0.5,
                vmax=cfg["data"]["num_phases"] - 0.5,
                interpolation="nearest",
            )
            panels[row, col].set_title(f"axis {axis}")
            panels[row, col].axis("off")
    fig.suptitle("Random training crops")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
