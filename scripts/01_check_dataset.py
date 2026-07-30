import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AXES, LabelPatchDataset, load_axis_paths
from src.train import load_train_config

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive.")

    cfg = load_train_config(args.config)
    paths = load_axis_paths(cfg.data.folder)
    figure, axes = plt.subplots(
        len(AXES),
        args.samples,
        squeeze=False,
        figsize=(3 * args.samples, 8),
    )
    for row, axis in enumerate(AXES):
        for column in range(args.samples):
            path = paths[axis][np.random.randint(len(paths[axis]))]
            dataset = LabelPatchDataset(
                (path,),
                crop_size=cfg.data.crop_size,
                patch_size=cfg.data.patch_size,
                num_phases=cfg.data.num_phases,
            )
            labels = dataset[0].numpy()
            axes[row, column].imshow(
                labels,
                cmap="gray",
                vmin=-0.5,
                vmax=cfg.data.num_phases - 0.5,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"axis {axis}")
            axes[row, column].axis("off")
    figure.suptitle("Random training crops")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
