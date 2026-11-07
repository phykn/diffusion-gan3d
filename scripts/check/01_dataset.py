import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check.common import (
    add_out_argument,
    add_seed_argument,
    prepare_output,
    show_labels,
)
from src.config.sample import build_stage_configs, find_sample, load_sample
from src.console import field, path_field, section
from src.data.crops import CropSampler
from src.export import export_config, read_export, resolve_export
from src.phase import PORE_PHASE
from src.plane import PLANE_DIRECTIONS, PLANES
from src.train.run import training_images

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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Show the same training crops at the LR and SR grids."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--weight",
        type=Path,
        help="Use the data saved with this generator.pt or run folder.",
    )
    source.add_argument(
        "--sample",
        default="default",
        help="Name in config/samples/ or a sample YAML path (default: default).",
    )
    add_out_argument(parser)
    add_seed_argument(parser)
    parser.add_argument(
        "--no-view", action="store_true", help="Save results without a viewer."
    )
    args = parser.parse_args(argv)
    section("Training crops")
    out = prepare_output(args, __file__)
    if args.weight:
        path = resolve_export(args.weight).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        stage = (
            "sr" if isinstance(payload, dict) and payload.get("stage") == "sr" else "lr"
        )
        config = export_config(read_export(payload, stage))
        name = path.parent.name
    else:
        path = find_sample(args.sample)
        config = build_stage_configs(load_sample(path))["lr"]
        name = path.stem
    path_field("Config", path)
    data = config.data
    planes = training_images(config)
    sizes = {"LR": data.lo_res_size, "SR": data.hi_res_size}
    samplers = {
        label: {
            PLANES.index(plane): CropSampler(
                images, data.crop_size, size, data.num_phases
            )
            for plane, images in planes.items()
        }
        for label, size in sizes.items()
    }
    crop = data.crop_size
    for label, size in sizes.items():
        field(label, f"{crop} px -> {size}x{size} (1 cell = {crop / size:g} px)")
    for plane, images in planes.items():
        field(plane, image_count(len(images.all_images())))
    fig = crop_figure(samplers, data.num_phases, args.seed)
    lr, sr = (f"{size}x{size}" for size in sizes.values())
    fig.suptitle(
        f"{name}: {crop} px crops, left LR {lr}, right SR {sr}",
        y=1 - 0.08 / fig.get_figheight(),
        va="top",
    )
    output = out / "crops.png"
    fig.savefig(output, dpi=150)
    path_field("Preview", output)
    if not args.no_view:
        plt.show()
    plt.close(fig)


def crop_figure(samplers: dict, num_phases: int, seed: int):
    """Rows of planes; each column pairs one random crop at the LR and SR grids."""
    axes = tuple(samplers["LR"])
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
        top = height - TITLE_HEIGHT - row * (FRACTION_HEIGHT + PANEL) - FRACTION_HEIGHT
        fig.text(
            LABEL_WIDTH / 2 / width,
            (top - PANEL / 2) / height,
            f"{plane}\n\n{down} ↓\n{right} →",
            ha="center",
            va="center",
        )
        for col in range(SAMPLES):
            # The same seed makes both samplers cut the same source region.
            crop_seed = int(rng.randint(2**31))
            images = {
                label: grid[axis]
                .sample(1, torch.Generator().manual_seed(crop_seed))
                .images[0]
                for label, grid in samplers.items()
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
