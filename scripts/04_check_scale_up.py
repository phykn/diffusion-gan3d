import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_generation_arguments,
    height_options,
    non_negative_int,
    positive_int,
    prepare_check,
    save_results,
)
from scripts.common.diagnostic import (
    anchor_accuracy,
    format_percent,
    middle_slices,
    select_display_index,
    select_indices,
    show_difference,
    show_labels,
    show_napari,
    show_slices,
)
from src.build.predict import load_generator
from src.console import field, metric, path_field, section, status
from src.device import check_device
from src.evaluate.label import phase_fractions, voxel_accuracy
from src.evaluate.seam import SeamQuality, measure_seams
from src.predict.anchor import PlaneAnchor
from src.predict.tiling.generator import TiledGenerator
from src.predict.tiling.layout import TilePlan
from src.predict.volume_condition import resolve_offset
from src.random import seeded_rng
from src.storage import load_volume

AXIS = 0


@dataclass(frozen=True)
class BaseResult:
    volume: torch.Tensor | None
    target: torch.Tensor | None
    indices: tuple[int, ...]
    accuracy: float | None


@dataclass(frozen=True)
class ScaleAssessment:
    quality: SeamQuality
    phase_fractions: torch.Tensor
    start: tuple[int, int, int] | None
    center: torch.Tensor | None
    base_match: float | None
    base_interior_match: float | None
    scaled_accuracy: float | None


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Inspect LR tiled generation.")
    parser.add_argument(
        "--gt",
        type=Path,
        help="Ground-truth TIFF volume used to build anchors.",
    )
    parser.add_argument(
        "--blocks",
        nargs=3,
        type=positive_int,
        default=(2, 2, 2),
        metavar=("D", "H", "W"),
        help="Number of output blocks along each axis (default: 2 2 2).",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int,
        help="Context added to each side of a block (default: config/gen.yaml).",
    )
    parser.add_argument(
        "--margin",
        type=non_negative_int,
        help="Voxels discarded from every outer face (default: model-derived).",
    )
    parser.add_argument(
        "--count",
        type=non_negative_int,
        help="Generate a base with this many anchor planes.",
    )
    add_generation_arguments(parser, anchors=True)
    args = parser.parse_args()
    return parser, args


def generate_base(
    args: argparse.Namespace, generator, anchor_count: int | None
) -> BaseResult:
    if anchor_count is None:
        field("Base", "none")
        return BaseResult(None, None, (), None)
    if anchor_count == 0:
        status("Generating unanchored base volume")
        base = generator.generate(
            guidance=args.guidance,
            domain=args.domain,
            **height_options(args),
            margin=args.margin,
        )
        return BaseResult(base, None, (), None)

    target = load_volume(args.gt)
    slices = target.movedim(AXIS, 0)
    indices = select_indices(slices.shape[0], anchor_count)
    anchors = tuple(
        PlaneAnchor(image=slices[index], axis=AXIS, index=index) for index in indices
    )
    status(f"Generating base with {len(indices)} anchor planes")
    base = generator.generate(
        anchors=anchors,
        anchor_strength=args.anchor_strength,
        guidance=args.guidance,
        domain=args.domain,
        **height_options(args),
        margin=args.margin,
    )
    return BaseResult(
        base,
        target,
        indices,
        anchor_accuracy(base, target, indices, AXIS),
    )


def assess_result(
    volume: torch.Tensor,
    base_result: BaseResult,
    stats: TilePlan,
    patch_size: int,
    num_phases: int,
) -> ScaleAssessment:
    quality = measure_seams(
        volume,
        stats.seams,
        stats.overlap,
        num_phases,
    )
    vfs = phase_fractions(volume, num_phases)
    base = base_result.volume
    if base is None:
        return ScaleAssessment(quality, vfs, None, None, None, None, None)

    start = resolve_offset(None, volume.shape, base.shape)
    region = tuple(slice(index, index + patch_size) for index in start)
    center = volume[region]
    shell = stats.base_shell
    interior = tuple(
        slice(shell, -shell)
        if volume.shape[axis] > patch_size and shell
        else slice(None)
        for axis in range(3)
    )
    scaled_accuracy = None
    if base_result.target is not None:
        scaled_accuracy = anchor_accuracy(
            center,
            base_result.target,
            base_result.indices,
            AXIS,
        )
    return ScaleAssessment(
        quality,
        vfs,
        start,
        center,
        voxel_accuracy(center, base),
        voxel_accuracy(center[interior], base[interior]),
        scaled_accuracy,
    )


def main() -> None:
    parser, args = parse_args()
    section("Scale-up check")
    settings = prepare_check(args, __file__)
    anchor_count = args.count if args.count is None or args.anchor_strength > 0.0 else 0
    if anchor_count is not None and anchor_count > 0 and args.gt is None:
        parser.error("--gt is required when --count is positive.")
    if args.gt is not None and not anchor_count:
        parser.error(
            "--gt requires active anchors (--count > 0 and --anchor-strength > 0)."
        )

    device = check_device(args.device)
    path_field("Weights", args.weight)

    generator = load_generator(args.weight, device=device)
    if anchor_count is not None and anchor_count > generator.patch_size:
        parser.error(f"--count must be at most {generator.patch_size}.")
    overlap = (
        min(settings.overlap, (generator.patch_size - 1) // 2)
        if args.overlap is None
        else args.overlap
    )
    if args.margin is None:
        args.margin = generator.default_margin
    scaled = TiledGenerator(generator)
    shape = scaled.shape_from_blocks(tuple(args.blocks), overlap)
    generation_shape = tuple(size + 2 * args.margin for size in shape)
    plan = scaled.plan(generation_shape, overlap)
    field(
        "Plan",
        f"{' × '.join(map(str, plan.grid))} tiles "
        f"({plan.tile_count} total), overlap {plan.overlap}, {device}",
    )
    field("Output", f"{' × '.join(map(str, shape))}, margin {args.margin}")
    with seeded_rng(args.seed, device):
        base_args = copy.copy(args)
        if generator.height_data is not None:
            generator.validate_height(
                shape, args.domain, args.height_origin, args.height_extent
            )
            base_args.height_origin += (
                (shape[0] - generator.patch_size)
                // 2
                * generator.height_data["crop_size"]
                / generator.height_data["lo_res_size"]
            )
        base_result = generate_base(base_args, generator, anchor_count)

        status("Scaling")

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        start = perf_counter()
        vol = scaled.generate(
            blocks=tuple(args.blocks),
            overlap=overlap,
            margin=args.margin,
            base=base_result.volume,
            vf=None,
            guidance=args.guidance,
            domain=args.domain,
            **height_options(args),
        )
    stats = scaled.stats
    elapsed = perf_counter() - start
    save_results(vol, args, generator.num_phases)
    assessment = assess_result(
        vol,
        base_result,
        stats,
        generator.patch_size,
        generator.num_phases,
    )
    quality = assessment.quality
    base, target = base_result.volume, base_result.target
    section("Quality")
    metric("Phase VF", format_phases(assessment.phase_fractions))
    metric("Seam TV", format_axes(quality.transition_tv))
    metric("Seam continuity", format_axes(quality.continuation_delta))
    if base is not None:
        metric("Base interior", format_percent(assessment.base_interior_match))
        if target is None:
            metric("Whole base", format_percent(assessment.base_match))
        else:
            metric(
                "Anchor match",
                f"{format_percent(base_result.accuracy)} → "
                f"{format_percent(assessment.scaled_accuracy)}",
            )

    performance = f"{elapsed:.1f} s"
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        performance += f", peak {peak:.2f} GiB"
    field("Elapsed", performance)

    if args.no_view:
        return
    if args.napari:
        show_napari(vol, name="scaled phases")
    elif base is not None:
        if target is None:
            show_unanchored_base_result(vol, base, assessment, generator.num_phases)
        else:
            show_base_result(
                vol,
                base,
                target,
                assessment,
                base_result.indices,
                AXIS,
                generator.num_phases,
            )
    else:
        show_slices(vol, generator.num_phases, "Joint tiled diffusion scale-up")


def show_unanchored_base_result(
    vol: torch.Tensor,
    base: torch.Tensor,
    assessment: ScaleAssessment,
    num_phases: int,
) -> None:
    idx = base.shape[0] // 2
    base_plane = base[idx]
    center_plane = assessment.center[idx]
    fig, panels = plt.subplots(2, 3, figsize=(11, 7))
    show_labels(panels[0, 0], base_plane, num_phases, "1. Base slice")
    show_labels(panels[0, 1], center_plane, num_phases, "2. Base area after scale")
    show_difference(panels[0, 2], center_plane != base_plane, "3. Scaled vs base")

    views = middle_slices(vol)
    for axis, (panel, img) in enumerate(zip(panels[1], views, strict=True)):
        plane_axes = tuple(value for value in range(3) if value != axis)
        top, left = (assessment.start[value] for value in plane_axes)
        height, width = (base.shape[value] for value in plane_axes)
        show_labels(panel, img, num_phases, f"{axis + 4}. Full volume axis {axis}")
        panel.add_patch(base_box(left - 0.5, top - 0.5, width, height))

    fig.suptitle(
        "Scale-up from unanchored base\n"
        f"cyan box: base {base.shape[0]}×{base.shape[1]}×{base.shape[2]} area"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def base_box(left: float, top: float, width: int, height: int) -> Rectangle:
    return Rectangle(
        (left, top), width, height, fill=False, edgecolor="#00a6d6", linewidth=1.5
    )


def show_base_result(
    vol: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    assessment: ScaleAssessment,
    indices: tuple[int, ...],
    axis: int,
    num_phases: int,
) -> None:
    idx = select_display_index(base.shape[axis], indices)
    target_plane = target.movedim(axis, 0)[idx]
    base_plane = base.movedim(axis, 0)[idx]
    center_plane = assessment.center.movedim(axis, 0)[idx]
    start = assessment.start
    full_plane = vol.movedim(axis, 0)[start[axis] + idx]
    fig, panels = plt.subplots(3, 3, figsize=(11, 10))
    plane_axes = tuple(value for value in range(3) if value != axis)
    full_height, full_width = (vol.shape[value] for value in plane_axes)
    top, left = (start[value] for value in plane_axes)
    height, width = base_plane.shape

    show_labels(panels[0, 0], target_plane, num_phases, "1. Reference anchor")
    show_labels(panels[0, 1], center_plane, num_phases, "2. Base area zoom")
    show_labels(panels[0, 2], full_plane, num_phases, "3. Full scaled slice")
    panels[0, 2].add_patch(base_box(left - 0.5, top - 0.5, width, height))

    show_difference(panels[1, 0], base_plane != target_plane, "4. Base vs reference")
    show_difference(panels[1, 1], center_plane != base_plane, "5. Scaled vs base")
    show_difference(
        panels[1, 2],
        center_plane != target_plane,
        "6. Scaled vs reference",
        extent=(left, left + width, top + height, top),
    )
    panels[1, 2].add_patch(base_box(left, top, width, height))
    panels[1, 2].axis("on")
    panels[1, 2].set_xlim(0, full_width)
    panels[1, 2].set_ylim(full_height, 0)
    panels[1, 2].set_facecolor("#e5e5e5")
    panels[1, 2].set_xticks(())
    panels[1, 2].set_yticks(())

    views = middle_slices(vol)
    for view_axis, (panel, img) in enumerate(zip(panels[2], views, strict=True)):
        show_labels(
            panel, img, num_phases, f"{view_axis + 7}. Full volume axis {view_axis}"
        )

    fig.suptitle(
        f"Scale-up from {len(indices)} anchor planes · displayed slice {idx}\n"
        f"cyan box: base {height}×{width} area · "
        f"full slice: {full_height}×{full_width}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def format_axes(
    values: tuple[float | None, float | None, float | None],
) -> str:
    return "  ".join(
        f"axis {axis}: {'n/a' if value is None else f'{value:.4f}'}"
        for axis, value in enumerate(values)
    )


def format_phases(values: torch.Tensor) -> str:
    return "  ".join(
        f"phase {phase}: {float(value):.2%}" for phase, value in enumerate(values)
    )


if __name__ == "__main__":
    main()
