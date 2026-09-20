import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import check_parser, resolve_weight, result_directory
from src.build.data import build_datasets
from src.config.files import find_train_config, load_yaml
from src.config.train import load_train_config
from src.plane import PLANES

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train" / "low_res.yaml"
SAMPLES = 4


def main() -> None:
    parser = check_parser(
        __file__,
        "Inspect training crops from saved weights or the current data preset.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--weight",
        type=Path,
        help="Use the data configuration saved with these weights.",
    )
    source.add_argument(
        "--config",
        type=Path,
        help="Training config; default: config/train/low_res.yaml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Preview PNG; default: a new directory under run/checks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument(
        "--domain",
        type=int,
        default=0,
        help="numeric domain ID (default: 0)",
    )
    args = parser.parse_args()
    if args.no_view:
        plt.switch_backend("Agg")
    config = (
        find_train_config(resolve_weight(args.weight))
        if args.weight
        else args.config or DEFAULT_CONFIG
    )
    stage = load_yaml(config).get("stage", "low_res")
    cfg = load_train_config(config, stage)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = build_datasets(cfg, high=stage == "sr")
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
            img = ds[path].argmax(0).numpy()
            panels[row, col].imshow(
                img,
                cmap="gray",
                vmin=-0.5,
                vmax=cfg["data"]["num_phases"] - 0.5,
                interpolation="nearest",
            )
            panels[row, col].set_title(PLANES[axis])
            panels[row, col].axis("off")
    fig.suptitle("Random training crops (dominant phase; training retains fractions)")
    fig.tight_layout()
    output = args.out or result_directory("01_check_dataset") / "crops.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Preview : {output.resolve()}")
    if not args.no_view:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
