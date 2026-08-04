"""Show random crops from each training axis."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import AXES
from src.build import build_datasets
from src.utils import load_yaml

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"
SAMPLES = 4


def main() -> None:
    cfg = load_yaml(DEFAULT_CONFIG)
    datasets = build_datasets(cfg)
    fig, panels = plt.subplots(
        len(AXES),
        SAMPLES,
        squeeze=False,
        figsize=(3 * SAMPLES, 8),
    )
    for row, axis in enumerate(AXES):
        ds = datasets[axis]
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
