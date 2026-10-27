import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.cli import add_domain_seed_arguments, result_directory
from scripts.common.diagnostic import show_labels
from src.build.data import build_datasets
from src.build.predict import resolve_weights
from src.config.files import find_saved_config, load_yaml
from src.config.sample import build_train_config, find_sample, load_sample
from src.config.train import load_train_config
from src.console import field, path_field, section
from src.phase import PORE_PHASE
from src.plane import PLANE_DIRECTIONS, PLANES

SAMPLES = 4
# Figure layout in inches.
PANEL = 1.9
PAIR_GAP = 0.05
CROP_GAP = 0.3
LABEL_WIDTH = 0.6
MARGIN = 0.1
TITLE_HEIGHT = 0.4
FRACTION_HEIGHT = 0.3
LEGEND_HEIGHT = 0.4


def phase_name(phase: int) -> str:
    return f"phase {phase} (pore)" if phase == PORE_PHASE else f"phase {phase}"


def image_count(count: int) -> str:
    return f"{count} image" if count == 1 else f"{count} images"


def phase_color(phase: int, num_phases: int):
    # The value show_labels maps phase to under its gray colormap.
    return plt.get_cmap("gray")((phase + 0.5) / num_phases)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the same training crops at the LR and SR grids."
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
        cfg = load_train_config(config, load_yaml(config)["stage"])
        name = config.parent.name
    else:
        config = find_sample(args.sample)
        cfg = load_train_config(
            build_train_config(load_sample(config), "low_res"), "low_res"
        )
        name = config.stem
    torch.manual_seed(args.seed)
    lr_domains = build_datasets(cfg)
    sr_domains = build_datasets(cfg, high_resolution=True)
    if args.domain not in lr_domains:
        parser.error(f"--domain must be one of {list(lr_domains)}")
    if len(lr_domains) > 1:
        name += f" (domain {args.domain})"
    stages = {"LR": lr_domains[args.domain], "SR": sr_domains[args.domain]}
    first = next(iter(stages["LR"]))
    crop = stages["LR"][first].crop_size
    sizes = {label: planes[first].patch_size for label, planes in stages.items()}
    section("Training crops")
    path_field("Config", config.resolve())
    for label, size in sizes.items():
        field(label, f"{crop} px -> {size}x{size} (1 cell = {crop / size:g} px)")
    for axis, dataset in stages["LR"].items():
        field(PLANES[axis], image_count(len(dataset)))
    fig = crop_figure(stages, cfg["data"]["num_phases"], args.seed)
    lr, sr = (f"{size}x{size}" for size in sizes.values())
    fig.suptitle(
        f"{name}: {crop} px crops, left LR {lr}, right SR {sr}",
        y=1 - 0.08 / fig.get_figheight(),
        va="top",
    )
    output = args.out or result_directory(__file__) / "crops.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    path_field("Preview", output.resolve())
    if not args.no_view:
        plt.show()
    plt.close(fig)


def crop_figure(stages: dict, num_phases: int, seed: int):
    """Rows of planes; each column pairs one random crop at the LR and SR grids."""
    axes = tuple(stages["LR"])
    # Fixed layout in inches: a narrow gap inside each LR/SR pair, a wider one
    # between pairs, and room above each pair for its phase fractions.
    pair = 2 * PANEL + PAIR_GAP
    width = LABEL_WIDTH + SAMPLES * pair + (SAMPLES - 1) * CROP_GAP + MARGIN
    height = TITLE_HEIGHT + len(axes) * (FRACTION_HEIGHT + PANEL) + LEGEND_HEIGHT
    fig = plt.figure(figsize=(width, height))
    rng = np.random.RandomState(seed)
    for row, axis in enumerate(axes):
        plane = PLANES[axis]
        down, right = PLANE_DIRECTIONS[plane]
        dataset = stages["LR"][axis]
        top = height - TITLE_HEIGHT - row * (FRACTION_HEIGHT + PANEL) - FRACTION_HEIGHT
        fig.text(
            LABEL_WIDTH / 2 / width,
            (top - PANEL / 2) / height,
            f"{plane}\n\n{down} ↓\n{right} →",
            ha="center",
            va="center",
        )
        for col in range(SAMPLES):
            group = dataset.path_groups[rng.randint(len(dataset.path_groups))]
            path = group[rng.randint(len(group))]
            # The same seed makes both datasets cut the same source region.
            crop_seed = rng.randint(2**31)
            images = {
                label: planes[axis].sample(path, np.random.RandomState(crop_seed))[
                    "image"
                ]
                for label, planes in stages.items()
            }
            left = LABEL_WIDTH + col * (pair + CROP_GAP)
            fig.text(
                (left + pair / 2) / width,
                (top + 0.05) / height,
                " / ".join(
                    f"{fraction:.0%}" for fraction in images["SR"].mean((1, 2)).tolist()
                ),
                ha="center",
                va="bottom",
                fontsize=9,
            )
            for index, image in enumerate(images.values()):
                panel = fig.add_axes(
                    (
                        (left + index * (PANEL + PAIR_GAP)) / width,
                        (top - PANEL) / height,
                        PANEL / width,
                        PANEL / height,
                    )
                )
                show_labels(panel, image.argmax(0), num_phases, "")
    fig.legend(
        handles=[
            Patch(color=phase_color(phase, num_phases), label=phase_name(phase))
            for phase in range(num_phases)
        ],
        loc="lower center",
        ncol=num_phases,
        frameon=False,
    )
    return fig


if __name__ == "__main__":
    main()
