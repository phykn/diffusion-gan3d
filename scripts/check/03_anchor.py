import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check.common import (
    add_generation_arguments,
    generation_options,
    middle_slices,
    non_negative_int,
    prepare_check,
    save_results,
    show_difference,
    show_labels,
    show_napari,
    write_json,
)
from src.api import LowResolutionAPI, PlaneAnchor
from src.console import field, metric, path_field, section, status
from src.data.images import read_labels
from src.stats.anchor import (
    measure_boundaries,
    measure_distance_divergence,
    measure_slice_smoothness,
)
from src.stats.labels import voxel_accuracy

DISPLAY_DISTANCES = (0, 1, 2, 4, 8, 16, 32, 64)


def select_indices(size: int, count: int) -> tuple[int, ...]:
    """count section indices spread evenly over size."""
    if count < 0 or count > size:
        raise ValueError("count must be between 0 and the volume depth.")
    return tuple((2 * index + 1) * size // (2 * count) for index in range(count))


def select_display_index(size: int, indices: tuple[int, ...]) -> int:
    center = size // 2
    if not indices:
        return center
    return min(indices, key=lambda index: (abs(index - center), index))


def anchor_accuracy(vol, target, indices: tuple[int, ...], axis: int):
    """Voxel accuracy on the anchored planes only."""
    if not indices:
        return None
    index = torch.tensor(indices, dtype=torch.long)
    actual = vol.movedim(axis, 0).index_select(0, index)
    expected = target.movedim(axis, 0).index_select(0, index)
    return voxel_accuracy(actual, expected)


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def format_profile(values, near_label: str) -> str:
    valid = [(d, value) for d, value in enumerate(values) if value is not None]
    if not valid:
        return "n/a"
    mean = sum(value for _, value in valid) / len(valid)
    distance, farthest = valid[-1]
    return (
        f"{near_label} {format_percent(valid[0][1])}, mean {format_percent(mean)}, "
        f"farthest {format_percent(farthest)} at distance {distance}"
    )


def print_anchor_effect(smoothness, profile, label: str) -> None:
    peak = ""
    if smoothness.peak_anchor_distance is not None:
        peak = f" at distance {smoothness.peak_anchor_distance}"
    metric(
        "Smoothness",
        f"p95 {format_ratio(smoothness.p95_ratio)}, "
        f"peak {format_ratio(smoothness.max_ratio)}{peak}",
    )
    if smoothness.reversal_rate is not None:
        metric(
            "Surface bumps",
            f"{smoothness.reversal_rate:.2%} vs baseline "
            f"{format_percent(smoothness.baseline_reversal_rate)} "
            f"({format_ratio(smoothness.reversal_ratio)})",
        )
    if profile:
        metric("Anchor effect", format_profile(profile, label))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Guide an LR sample with anchor planes: sections of a "
        "generated reference, or one boundary plane from a real image."
    )
    add_generation_arguments(parser, anchors=True, origin_default=None)
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
        help="Evenly spread planes of a generated reference (default: 3).",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        help="Real 2D label PNG; its center crop anchors one boundary plane "
        "instead of --count reference planes.",
    )
    parser.add_argument(
        "--side",
        choices=("start", "end"),
        help="Boundary plane of --anchor (default: start).",
    )
    args = parser.parse_args(argv)
    if args.side is not None and args.anchor is None:
        parser.error("--side needs --anchor.")
    if args.anchor is not None:
        args.side = args.side or "start"
    return parser, args


def main(argv=None) -> None:
    parser, args = parse_args(argv)
    section("Anchor check")
    prepare_check(args, __file__)
    api = LowResolutionAPI(args.weight, args.device)
    patch = api.input_size
    if args.anchor is None:
        count = args.count if args.anchor_strength > 0 else 0
        if count > patch:
            parser.error(f"--count must be at most {patch}.")
        status("Generating reference")
        reference = api.generate(seed=args.seed, **generation_options(args))
        sections = reference.movedim(args.axis, 0)
        indices = select_indices(sections.shape[0], count)
        anchors = [PlaneAnchor(sections[index], args.axis, index) for index in indices]
        field("Anchors", f"{len(indices)} planes on axis {args.axis}")
        target, labels, source = reference, None, None
    else:
        index = 0 if args.side == "start" else patch - 1
        image, crop, height = load_anchor_image(args.anchor, api)
        # A real xz/yz image fixes its own depth unless the options set it.
        if api.config.conditioning.height_enabled and args.axis in (1, 2):
            if args.height_origin is None:
                args.height_origin = float(crop[0])
            if args.height_extent is None:
                args.height_extent = float(height)
        indices = (index,)
        anchors = [PlaneAnchor(image, args.axis, index)]
        labels = image.argmax(0)
        source = {"path": str(args.anchor.resolve()), "crop": list(crop)}
        field(
            "Anchor",
            f"{args.anchor.resolve()}, crop {crop} -> input {patch} x {patch}",
        )
        field("Boundary", f"axis {args.axis}, {args.side} plane {index}")
        target = None
    if indices:
        field("Strength", f"{args.anchor_strength:.2f}")
    options = generation_options(args) | {"seed": args.seed + 1}
    status("Generating anchored sample")
    generated = api.generate(anchors, anchor_strength=args.anchor_strength, **options)
    # The same noise without anchors shows what the anchors changed.
    baseline = api.generate(**options)
    save_results(generated, args, api.num_phases)

    profile = measure_distance_divergence(generated, baseline, indices, args.axis)
    smoothness = measure_slice_smoothness(generated, indices, args.axis, baseline)
    if labels is None:
        match = anchor_accuracy(generated, target, indices, args.axis)
    else:
        match = voxel_accuracy(generated.movedim(args.axis, 0)[indices[0]], labels)
    section("Quality")
    metric("Anchor match", format_percent(match))
    report = {
        "axis": args.axis,
        "indices": list(indices),
        "anchor_strength": args.anchor_strength,
        "anchor": source,
        "side": args.side,
        "anchor_match": match,
        "smoothness": dataclasses.asdict(smoothness),
        "divergence": list(profile),
    }
    if labels is not None:
        boundary = measure_boundaries(generated, indices, args.axis)
        metric(
            "First change",
            f"{format_percent(boundary.anchor_change)} vs ordinary "
            f"{format_percent(boundary.ordinary_change)} "
            f"({format_ratio(boundary.change_ratio)})",
        )
        report["boundary"] = dataclasses.asdict(boundary)
    print_anchor_effect(smoothness, profile, "anchor" if labels is None else "boundary")
    path_field("Report", write_json(args.out / "anchor.json", report))
    show_view = not args.no_view and not args.napari
    if labels is not None:
        render_strip(
            generated,
            labels,
            args.axis,
            args.side,
            api.num_phases,
            args.out / "volume_continuation.png",
            show=show_view,
        )
    elif show_view:
        show_result(generated, target, indices, args.axis, api.num_phases)
    if args.napari and not args.no_view:
        show_napari(generated)


def load_anchor_image(path: Path, api: LowResolutionAPI):
    """Center crop, resized to the LR grid; also the crop and image height."""
    values = read_labels(path)
    height, width = values.shape
    size = api.crop_size
    if height < size or width < size:
        raise ValueError(f"anchor image must contain a {size} x {size} crop.")
    top, left = (height - size) // 2, (width - size) // 2
    crop = torch.from_numpy(values[top : top + size, left : left + size].copy())
    return api.prepare_image(crop.long()), (top, left, size, size), height


def show_result(vol, target, indices, axis: int, num_phases: int) -> None:
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

    views = middle_slices(vol)
    orth_axes = tuple(value for value in (0, 1, 2) if value != axis)
    for panel, view_axis in zip(panels[2, :2], orth_axes, strict=True):
        view = views[view_axis]
        # Put the anchor axis on rows so anchor planes show as horizontal lines.
        anchor_dim = axis if axis < view_axis else axis - 1
        oriented = view if anchor_dim == 0 else view.T
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


def render_strip(volume, anchor, axis, side, num_phases, output: Path, show: bool):
    slices = volume.movedim(axis, 0)
    if side == "end":
        slices = slices.flip(0)
    last = len(slices) - 1
    distances = tuple(distance for distance in DISPLAY_DISTANCES if distance < last)
    distances += (last,)
    images = (anchor,) + tuple(slices[distance] for distance in distances)
    index = 0 if side == "start" else last
    titles = (f"Condition input (target)\naxis {axis}, index {index}",) + tuple(
        (
            f"Generated output\naxis {axis}, index {index}"
            if distance == 0
            else f"Generated d={distance}"
        )
        for distance in distances
    )
    columns = 5
    rows = (len(images) + columns - 1) // columns
    figure, panels = plt.subplots(
        rows, columns, squeeze=False, figsize=(12, 2.5 * rows)
    )
    cmap = plt.get_cmap("gray", num_phases)
    for panel, image, title in zip(panels.flat, images, titles, strict=False):
        panel.imshow(
            image.numpy(),
            cmap=cmap,
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")
    panels.flat[0].add_patch(
        Rectangle(
            (-0.5, -0.5),
            anchor.shape[1],
            anchor.shape[0],
            fill=False,
            edgecolor="#F97316",
            linewidth=3.0,
        )
    )
    for panel in tuple(panels.flat)[len(images) :]:
        panel.axis("off")
    figure.suptitle(f"One-sided continuation from the {side} plane")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    path_field("Figure", output)
    if show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
