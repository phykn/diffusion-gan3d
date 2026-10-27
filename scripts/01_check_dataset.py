import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import add_domain_seed_arguments, result_directory
from scripts.common.diagnostic import show_labels
from src.build.data import build_datasets
from src.build.predict import resolve_weights
from src.config.files import find_saved_config, load_yaml
from src.config.sample import build_train_config, find_sample, load_sample
from src.config.train import load_train_config
from src.phase import PORE_PHASE
from src.plane import PLANE_DIRECTIONS, PLANES

SAMPLES = 4


def phase_name(phase: int) -> str:
    return f"phase {phase} (pore)" if phase == PORE_PHASE else f"phase {phase}"


def image_count(count: int) -> str:
    return f"{count} image" if count == 1 else f"{count} images"


def phase_color(phase: int, num_phases: int):
    # The value show_labels maps phase to under its gray colormap.
    return plt.get_cmap("gray")((phase + 0.5) / num_phases)


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
    num_phases = cfg["data"]["num_phases"]
    planes = datasets[args.domain]
    first = next(iter(planes.values()))
    name = config.parent.name if args.weight else config.stem
    stage_label = "SR (HR)" if stage == "sr" else "LR"
    voxel = f"{first.crop_size} px crop -> {first.patch_size}x{first.patch_size} {stage_label} grid"
    print(f"Config  : {config.resolve()}")
    print(f"Crops   : {voxel} (1 cell = {first.crop_size / first.patch_size:g} px)")
    for axis, ds in planes.items():
        print(f"{PLANES[axis]:<8}: {image_count(len(ds))}")
    fig, panels = plt.subplots(
        len(planes),
        SAMPLES,
        squeeze=False,
        figsize=(3 * SAMPLES, 2.9 * len(planes) + 0.6),
    )
    for row, (axis, ds) in enumerate(planes.items()):
        plane = PLANES[axis]
        down, right = PLANE_DIRECTIONS[plane]
        panels[row, 0].text(
            -0.08,
            0.5,
            f"{plane}  ({image_count(len(ds))})\nrows {down}, columns {right}",
            transform=panels[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
        )
        for col in range(SAMPLES):
            group = ds.path_groups[np.random.randint(len(ds.path_groups))]
            path = group[np.random.randint(len(group))]
            image = ds[path]["image"]
            fractions = "  ".join(
                f"{phase}: {fraction:.0%}"
                for phase, fraction in enumerate(image.mean((1, 2)).tolist())
            )
            show_labels(
                panels[row, col],
                image.argmax(0),
                num_phases,
                f"{path.name}\n{fractions}",
            )
            panels[row, col].title.set_fontsize(9)
    fig.legend(
        handles=[
            Patch(color=phase_color(phase, num_phases), label=phase_name(phase))
            for phase in range(num_phases)
        ],
        loc="lower center",
        ncol=num_phases,
        frameon=False,
    )
    fig.suptitle(
        f"{name}, domain {args.domain}: random training crops, {voxel}\n"
        "Each cell shows its dominant phase; titles give phase fractions."
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 1))
    output = args.out or result_directory(__file__) / "crops.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Preview : {output.resolve()}")
    if not args.no_view:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
