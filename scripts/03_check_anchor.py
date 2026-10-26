import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_generation_arguments,
    height_options,
    non_negative_int,
    prepare_check,
    save_results,
)
from scripts.common.diagnostic import (
    anchor_accuracy,
    generate_with_baseline,
    middle_slices,
    print_anchor_effect,
    select_display_index,
    select_indices,
    show_difference,
    show_labels,
    show_napari,
)
from src.build.predict import load_generator
from src.device import check_device
from src.evaluate.anchor import measure_distance_divergence, measure_slice_smoothness
from src.evaluate.label import voxel_accuracy
from src.predict.anchor import PlaneAnchor
from src.random import seeded_rng


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect how generated planes constrain a second LR sample."
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Anchor plane: 0=xy, 1=xz, 2=yz (default: 0).",
    )
    parser.add_argument(
        "--count",
        type=non_negative_int,
        default=3,
        help="Number of evenly distributed anchor planes (default: 3).",
    )
    add_generation_arguments(parser, anchors=True)
    args = parser.parse_args()
    prepare_check(args, __file__)
    anchor_count = args.count if args.anchor_strength > 0.0 else 0

    device = check_device(args.device)
    print(f"\nWeights : {args.weight}")

    generator = load_generator(args.weight, device=device)
    if anchor_count > generator.patch_size:
        parser.error(f"--count must be at most {generator.patch_size}.")
    with seeded_rng(args.seed, device):
        print("Generating reference...", flush=True)
        target = generator.generate(
            anchors=(),
            anchor_strength=0.0,
            guidance=args.guidance,
            domain=args.domain,
            **height_options(args),
            margin=generator.default_margin,
        )
        target_slices = target.movedim(args.axis, 0)
        indices = select_indices(target_slices.shape[0], anchor_count)
        anchors = tuple(
            PlaneAnchor(image=target_slices[index], axis=args.axis, index=index)
            for index in indices
        )
        print(f"Input   : {' x '.join(map(str, target.shape))}, {device}")
        print(f"Anchors : {len(indices)} planes on axis {args.axis}")
        if indices:
            print(f"Strength: {args.anchor_strength:.2f}")
        print("Generating anchored sample...", flush=True)

        gen, baseline = generate_with_baseline(
            generator,
            anchors,
            args.anchor_strength,
            guidance=args.guidance,
            domain=args.domain,
            margin=generator.default_margin,
            **height_options(args),
        )
    save_results(gen, args, generator.num_phases)
    profile = measure_distance_divergence(gen, baseline, indices, args.axis)
    smoothness = measure_slice_smoothness(gen, indices, args.axis, baseline)
    print("\nQuality")
    score = anchor_accuracy(gen, target, indices, args.axis)
    print(f"Anchor match  : {'n/a' if score is None else f'{score:7.2%}'}")
    print_anchor_effect(smoothness, profile, "anchor")
    if args.no_view:
        return
    if args.napari:
        show_napari(gen)
    else:
        show_result(gen, target, indices, args.axis, generator.num_phases)


def show_result(
    vol: torch.Tensor,
    target: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    num_phases: int,
) -> None:
    slices = vol.movedim(axis, 0)
    index = select_display_index(slices.shape[0], indices)
    target_plane = target.movedim(axis, 0)[index]
    gen_plane = slices[index]
    accuracy = voxel_accuracy(gen_plane, target_plane)
    fig, panels = plt.subplots(3, 3, figsize=(10, 9))

    reference = "1. Input anchor" if indices else "1. Reference center"
    show_labels(panels[0, 0], target_plane, num_phases, reference)
    show_labels(panels[0, 1], gen_plane, num_phases, "2. Anchor-guided output")
    show_difference(
        panels[0, 2], gen_plane != target_plane, "3. Difference (red = mismatch)"
    )

    before = max(0, index - 1)
    after = min(slices.shape[0] - 1, index + 1)
    neighbors = (
        (before, f"4. Before (slice {before})"),
        (index, f"5. Anchor position (slice {index})"),
        (after, f"6. After (slice {after})"),
    )
    for panel, (slice_idx, title) in zip(panels[1], neighbors, strict=True):
        show_labels(panel, slices[slice_idx], num_phases, title)

    orth_axes = tuple(value for value in (0, 1, 2) if value != axis)
    for panel, view_axis in zip(panels[2, :2], orth_axes, strict=True):
        section = middle_slices(vol)[view_axis]
        anchor_dim = axis if axis < view_axis else axis - 1
        oriented = section if anchor_dim == 0 else section.T
        show_labels(
            panel, oriented, num_phases, f"Orthogonal view from axis {view_axis}"
        )

    anchor_map = panels[2, 2]
    anchor_map.imshow(
        np.ones((slices.shape[0], slices.shape[-1]), dtype=np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    for anchor_idx in indices:
        anchor_map.axhline(anchor_idx, color="#e63946", linewidth=1.5)
    anchor_map.set_title("9. Anchor positions (red)")
    anchor_map.axis("off")

    mode = "Distributed anchors" if indices else "Unanchored baseline"
    score_label = "anchor" if indices else "reference center"
    fig.suptitle(
        f"{mode} · {score_label} {100 * accuracy:.1f}% matched\n"
        f"{len(indices)} planes · axis {axis} · slice {index}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


if __name__ == "__main__":
    main()
