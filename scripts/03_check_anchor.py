import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_generation_arguments,
    check_parser,
    height_options,
    non_negative_int,
    prepare_check,
    save_preview,
)
from scripts.common.diagnostic import (
    anchor_accuracy,
    format_percent,
    format_profile,
    format_ratio,
    generate_with_baseline,
    middle_slices,
    select_display_index,
    select_indices,
    show_napari,
)
from src.anchor import PlaneAnchor
from src.build.predict import load_generator
from src.device import check_device
from src.evaluate.anchor import (
    SliceSmoothness,
    measure_distance_divergence,
    measure_slice_smoothness,
)
from src.evaluate.label import voxel_accuracy
from src.random import seeded_rng
from src.storage import save_volume


def main() -> None:
    parser = check_parser(
        __file__, "Inspect how generated planes constrain a second LR sample."
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="anchor plane: 0=xy, 1=xz, 2=yz",
    )
    parser.add_argument(
        "--count",
        type=non_negative_int,
        default=3,
        help="number of evenly distributed anchor planes to use (default: 3)",
    )
    add_generation_arguments(parser, anchors=True)
    args = parser.parse_args()
    prepare_check(args, __file__)
    anchor_count = args.count if args.anchor_strength > 0.0 else 0

    device = check_device(args.device)
    print(f"\nWeights : {args.weight.resolve()}")

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
            print(f"Guidance : strength {args.anchor_strength:.2f}")
        print("Generating conditioned sample...", flush=True)

        gen, baseline = generate_with_baseline(
            generator,
            anchors,
            args.anchor_strength,
            guidance=args.guidance,
            domain=args.domain,
            margin=generator.default_margin,
            **height_options(args),
        )
    save_volume(gen, args.out)
    print(f"Saved   : {args.out.resolve()}", flush=True)
    save_preview(gen, args, generator.num_phases)
    gen_slices = gen.movedim(args.axis, 0)
    anchor_acc = anchor_accuracy(gen, target, indices, args.axis)
    index = select_display_index(generator.patch_size, indices)
    target_plane = target_slices[index]
    gen_plane = gen_slices[index]
    mismatch = gen_plane != target_plane
    slice_acc = voxel_accuracy(gen_plane, target_plane)
    baseline_profile = measure_distance_divergence(
        gen,
        baseline,
        indices,
        args.axis,
        generator.patch_size - 1,
    )
    smoothness = measure_slice_smoothness(gen, indices, args.axis, baseline)

    print_quality(anchor_acc, baseline_profile, smoothness)
    if args.no_view:
        return
    if args.napari:
        show_napari(gen)
    else:
        show_result(
            gen,
            target_plane,
            gen_plane,
            mismatch,
            indices=indices,
            axis=args.axis,
            index=index,
            num_phases=generator.num_phases,
            accuracy=slice_acc,
        )


def print_quality(
    anchor_acc: float | None,
    baseline_profile: tuple[float | None, ...],
    smoothness: SliceSmoothness,
) -> None:
    print("\nQuality")
    score = "n/a" if anchor_acc is None else f"{anchor_acc:7.2%}"
    print(f"Anchor match  : {score}")
    distance = (
        "n/a"
        if smoothness.peak_anchor_distance is None
        else str(smoothness.peak_anchor_distance)
    )
    print(
        "Smoothness    : "
        f"p95 {format_ratio(smoothness.p95_ratio)}, "
        f"peak {format_ratio(smoothness.max_ratio)} at distance {distance}"
    )
    if smoothness.reversal_rate is not None:
        print(
            "Surface bumps : "
            f"{smoothness.reversal_rate:.2%} vs baseline "
            f"{format_percent(smoothness.baseline_reversal_rate)} "
            f"({format_ratio(smoothness.reversal_ratio)})"
        )
    if baseline_profile:
        print(f"Anchor effect  : {format_profile(baseline_profile, 'anchor')}")


def show_result(
    vol: torch.Tensor,
    target: torch.Tensor,
    gen: torch.Tensor,
    mismatch: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    index: int,
    num_phases: int,
    accuracy: float,
) -> None:
    slices = vol.movedim(axis, 0)
    cmap = "gray"
    diff_cmap = ListedColormap(("#f7f7f7", "#e63946"))
    fig, panels = plt.subplots(3, 3, figsize=(10, 9))

    comparisons = (
        (target, "1. Input anchor" if indices else "1. Reference center", False),
        (gen, "2. Anchor-conditioned output", False),
        (mismatch, "3. Difference (red = mismatch)", True),
    )
    for panel, (img, title, difference) in zip(
        panels[0],
        comparisons,
        strict=True,
    ):
        panel.imshow(
            img.numpy(),
            cmap=diff_cmap if difference else cmap,
            vmin=0 if difference else -0.5,
            vmax=1 if difference else num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")

    before = max(0, index - 1)
    after = min(slices.shape[0] - 1, index + 1)
    neighbors = (
        (before, f"4. Before (slice {before})"),
        (index, f"5. Anchor position (slice {index})"),
        (after, f"6. After (slice {after})"),
    )
    for panel, (slice_idx, title) in zip(
        panels[1],
        neighbors,
        strict=True,
    ):
        panel.imshow(
            slices[slice_idx].numpy(),
            cmap=cmap,
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")

    orth_axes = tuple(value for value in (0, 1, 2) if value != axis)
    for panel, view_axis in zip(
        panels[2, :2],
        orth_axes,
        strict=True,
    ):
        section = middle_slices(vol)[view_axis]
        anchor_dim = axis if axis < view_axis else axis - 1
        oriented = section if anchor_dim == 0 else section.T
        panel.imshow(
            oriented.numpy(),
            cmap=cmap,
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(f"Orthogonal view from axis {view_axis}")
        panel.axis("off")

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

    mode = (
        "Distributed learned conditional anchors" if indices else "Unanchored baseline"
    )
    score_label = "anchor" if indices else "reference center"
    fig.suptitle(
        f"{mode} · {score_label} {100 * accuracy:.1f}% matched\n"
        f"{len(indices)} planes · axis {axis} · slice {index}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


if __name__ == "__main__":
    main()
