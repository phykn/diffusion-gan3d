import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import (
    continuation_delta,
    phase_change_rate,
    phase_fractions,
    transition_counts,
    transition_tv,
    voxel_accuracy,
)
from src.scale import ScaledGenerator, ScalePlan

AXIS = 0


@dataclass(frozen=True)
class SeamQuality:
    change_ratio: tuple[float | None, float | None, float | None]
    transition_tv: tuple[float | None, float | None, float | None]
    continuation_delta: tuple[float | None, float | None, float | None]


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
    center: torch.Tensor | None
    base_match: float | None
    base_interior_match: float | None
    scaled_accuracy: float | None
    interior_quality: SeamQuality | None


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace, int | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="generator weight to load",
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=0,
        help="numeric domain ID (default: 0)",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        help="ground-truth TIFF volume used to build anchors",
    )
    parser.add_argument(
        "--blocks",
        nargs=3,
        type=positive_int,
        default=(2, 2, 2),
        metavar=("D", "H", "W"),
        help="number of output blocks along each axis (default: 2 2 2)",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int,
        help="context added to each side of a block (default: config/gen.yaml)",
    )
    parser.add_argument(
        "--crop-margin",
        type=non_negative_int,
        help="voxels discarded from every outer face (default: config/gen.yaml)",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="generate a base with this many anchor planes",
    )
    parser.add_argument(
        "--anchor-strength",
        type=unit_interval,
        default=1.0,
        help="base anchor conditioning strength from 0 to 1 (default: 1)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="classifier-free guidance scale (default: config/gen.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="optional output path for the generated TIFF volume",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="show the complete scaled phase volume in Napari",
    )
    args = parser.parse_args()
    if args.count is not None and args.count < 0:
        parser.error("--count must be non-negative.")
    anchor_count = args.count if args.count is None or args.anchor_strength > 0.0 else 0
    if anchor_count is not None and anchor_count > 0 and args.gt is None:
        parser.error("--gt is required when --count is positive.")
    if args.gt is not None and not anchor_count:
        parser.error(
            "--gt requires active anchors (--count > 0 and --anchor-strength > 0)."
        )
    return parser, args, anchor_count


def generate_base(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    generator,
    anchor_count: int | None,
) -> BaseResult:
    if anchor_count is not None and anchor_count > generator.patch_size:
        parser.error(f"--count must be at most {generator.patch_size}.")
    if anchor_count is None:
        print("Base       : none")
        return BaseResult(None, None, (), None)
    if anchor_count == 0:
        print("Base       : unanchored")
        print("Status     : generating base...", flush=True)
        base = generator.generate(
            guidance_scale=args.guidance_scale,
            domain=args.domain,
        )
        return BaseResult(base, None, (), None)

    assert args.gt is not None
    target = load_volume(
        args.gt,
        generator.patch_size,
        generator.num_phases,
    )
    slices = target.movedim(AXIS, 0)
    indices = select_indices(slices.shape[0], anchor_count)
    anchors = tuple(
        PlaneAnchor(image=slices[index], axis=AXIS, index=index) for index in indices
    )
    print(f"Base       : {len(indices)} anchor planes")
    print(f"GT         : {args.gt.resolve()}")
    print("Status     : generating base...", flush=True)
    base = generator.generate(
        anchors=anchors,
        anchor_strength=args.anchor_strength,
        guidance_scale=args.guidance_scale,
        domain=args.domain,
    )
    return BaseResult(
        base,
        target,
        indices,
        get_accuracy(base, target, indices, AXIS),
    )


def assess_result(
    volume: torch.Tensor,
    base_result: BaseResult,
    stats: ScalePlan,
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

    start = tuple((size - patch_size) // 2 for size in volume.shape)
    region = tuple(slice(index, index + patch_size) for index in start)
    center = volume[region]
    base_match = float((center == base).to(torch.float32).mean())
    shell = stats.base_shell
    interior = tuple(
        slice(shell, -shell)
        if volume.shape[axis] > patch_size and shell
        else slice(None)
        for axis in range(3)
    )
    base_interior_match = float(
        (center[interior] == base[interior]).to(torch.float32).mean()
    )
    scaled_accuracy = None
    if base_result.target is not None:
        scaled_accuracy = get_accuracy(
            center,
            base_result.target,
            base_result.indices,
            AXIS,
        )
    boundaries = tuple(
        (
            tuple(
                index
                for index in (
                    start[axis] + shell,
                    region[axis].stop - shell,
                )
                if 0 < index < volume.shape[axis]
            )
            if volume.shape[axis] > patch_size
            else ()
        )
        for axis in range(3)
    )
    interior_quality = measure_seams(
        volume,
        boundaries,
        stats.overlap,
        num_phases,
    )
    return ScaleAssessment(
        quality,
        vfs,
        center,
        base_match,
        base_interior_match,
        scaled_accuracy,
        interior_quality,
    )


def main() -> None:
    parser, args, anchor_count = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = args.weight
    print("\nScale-up generation")
    print("-------------------")
    print(f"Weight     : {weight.resolve()}", flush=True)
    if args.gt is not None:
        print(f"GT         : {args.gt.resolve()}")

    generator = load_generator(weight, device=device)
    settings = load_generation_settings()
    overlap = settings.overlap if args.overlap is None else args.overlap
    crop_margin = settings.crop_margin if args.crop_margin is None else args.crop_margin
    guidance_scale = (
        settings.guidance_scale if args.guidance_scale is None else args.guidance_scale
    )
    args.guidance_scale = guidance_scale
    scaled = ScaledGenerator(generator)
    shape = scaled.shape_from_blocks(tuple(args.blocks), overlap)
    generation_shape = tuple(size + 2 * crop_margin for size in shape)
    plan = scaled.plan(generation_shape, overlap)
    print_plan(plan, device)
    print(f"Output shape : {' × '.join(map(str, shape))}")
    print(f"Crop margin  : {crop_margin} per outer face")
    base_result = generate_base(parser, args, generator, anchor_count)
    base = base_result.volume
    target = base_result.target
    indices = base_result.indices
    base_acc = base_result.accuracy

    conditioning = "soft base" if base is not None else "none"
    print(f"Conditioning : {conditioning}")
    postprocess = (
        f"center crop ({crop_margin} voxels per outer face)" if crop_margin else "none"
    )
    print(f"Postprocess  : {postprocess}")

    print("Status     : scaling...", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = perf_counter()
    vol = scaled.generate(
        blocks=tuple(args.blocks),
        overlap=overlap,
        crop_margin=crop_margin,
        base=base,
        vf=None,
        guidance_scale=guidance_scale,
        domain=args.domain,
    )
    stats = scaled.stats
    assert stats is not None
    elapsed = perf_counter() - start
    print("Status     : complete", flush=True)
    if args.out is not None:
        save_volume(vol, args.out)
    assessment = assess_result(
        vol,
        base_result,
        stats,
        generator.patch_size,
        generator.num_phases,
    )
    quality = assessment.quality
    vfs = assessment.phase_fractions
    scaled_acc = assessment.scaled_accuracy
    base_match = assessment.base_match
    base_interior_match = assessment.base_interior_match
    interior_quality = assessment.interior_quality
    center = assessment.center
    print("\nQuality")
    print("-------")
    print(f"Phase VF    : {format_phases(vfs)}")
    print(f"Seam change ratio  : {format_axes(quality.change_ratio)}")
    print(f"Transition TV      : {format_axes(quality.transition_tv)}")
    print(f"Continuation delta : {format_axes(quality.continuation_delta)}")
    if base is not None:
        print("\nBase")
        print("----")
        print(f"Base interior match : {format_score(base_interior_match)}")
        if target is None:
            print(f"Whole base match    : {format_score(base_match)}")
        else:
            print(f"Anchor match before : {format_score(base_acc)}")
            print(f"Anchor match after  : {format_score(scaled_acc)}")
        assert interior_quality is not None
        print(f"Base boundary change : {format_axes(interior_quality.change_ratio)}")
        print(f"Base boundary TV     : {format_axes(interior_quality.transition_tv)}")
        print(
            f"Base continuity      : {format_axes(interior_quality.continuation_delta)}"
        )

    print("\nPerformance")
    print("-----------")
    print(f"Elapsed     : {elapsed:.1f} s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"Peak memory : {peak:.2f} GiB")

    if args.napari:
        show_napari(vol)
    elif base is not None and center is not None:
        if target is None:
            show_unanchored_base_result(
                vol,
                base,
                center,
                generator.num_phases,
            )
        else:
            show_base_result(
                vol,
                base,
                target,
                center,
                indices,
                AXIS,
                generator.num_phases,
            )
    else:
        show_slices(vol, generator.num_phases)


def save_volume(vol: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, vol.detach().cpu().to(torch.uint8).numpy())
    print(f"Output     : {path.resolve()}", flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def print_plan(plan: ScalePlan, device: torch.device) -> None:
    print(f"Output shape : {' × '.join(map(str, plan.shape))}")
    print(f"Blocks       : {' × '.join(map(str, plan.grid))}")
    print(f"Block count  : {plan.tile_count}")
    print(f"Block size   : {plan.tile_size}")
    print(f"Stride       : {plan.stride}")
    print(f"Margin       : {plan.overlap} per shared face")
    print("Boundaries   : " + " × ".join(str(len(axis)) for axis in plan.seams))
    print(f"State memory : {format_bytes(plan.states_bytes)}")
    print(f"Fusion memory: {format_bytes(plan.fusion_bytes)}")
    print(f"Input memory : {format_bytes(plan.tile_bytes)}")
    print(f"Workspace    : {format_bytes(plan.workspace_bytes)}")
    print(f"CUDA total   : {format_bytes(plan.cuda_bytes)}")
    print(f"CPU total    : {format_bytes(plan.cpu_bytes)}")
    print(f"Output size  : {format_bytes(plan.output_bytes)}")
    print(f"Device       : {device}")


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise RuntimeError("unreachable")


def load_volume(path: Path, patch_size: int, num_phases: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"anchor volume was not found: {path}")
    vol = np.asarray(tifffile.imread(path))
    if vol.ndim != 3 or vol.size == 0:
        raise ValueError("anchor volume must be a non-empty 3D array.")
    if vol.dtype != np.uint8:
        raise ValueError(f"anchor volume must contain uint8 phases: {path}")
    if int(vol.max()) >= num_phases:
        raise ValueError(
            f"anchor volume must contain phases from 0 to {num_phases - 1}."
        )

    tensor = torch.from_numpy(np.array(vol, copy=True)).long()
    if tensor.shape != (patch_size, patch_size, patch_size):
        tensor = F.interpolate(
            tensor[None, None].to(torch.float32),
            size=(patch_size, patch_size, patch_size),
            mode="nearest",
        )[0, 0].to(torch.long)
    return tensor


def select_indices(size: int, count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    return tuple((2 * idx + 1) * size // (2 * count) for idx in range(count))


def select_display_index(size: int, indices: tuple[int, ...]) -> int:
    center = size // 2
    if not indices:
        return center
    return min(indices, key=lambda idx: (abs(idx - center), idx))


def get_accuracy(
    vol: torch.Tensor,
    target: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
) -> float | None:
    if not indices:
        return None
    idx = torch.tensor(indices, dtype=torch.long)
    actual = vol.movedim(axis, 0).index_select(0, idx)
    expected = target.movedim(axis, 0).index_select(0, idx)
    return voxel_accuracy(actual, expected)


def measure_seams(
    vol: torch.Tensor,
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    band_size: int,
    num_phases: int,
) -> SeamQuality:
    changes = []
    tvs = []
    deltas = []
    for axis, positions in enumerate(seams):
        if not positions:
            changes.append(None)
            tvs.append(None)
            deltas.append(None)
            continue
        pair_count = vol.shape[axis] - 1
        width = max(1, min(band_size, 4))
        band_idx = sorted(
            {
                idx
                for pos in positions
                for idx in range(
                    max(0, pos - width),
                    min(pair_count, pos + width),
                )
            }
        )
        band_set = set(band_idx)
        inner_idx = [idx for idx in range(pair_count) if idx not in band_set]
        if len(inner_idx) > 64:
            selected = np.linspace(0, len(inner_idx) - 1, num=64, dtype=int)
            inner_idx = [inner_idx[idx] for idx in selected]
        selected_idx = sorted(band_set | set(inner_idx))
        inner_set = set(inner_idx)
        inner_rates = []
        band_rates = {}
        inner_counts = torch.zeros(
            num_phases,
            num_phases,
            dtype=torch.float64,
        )
        band_counts = {}
        for idx in selected_idx:
            prev = vol.select(axis, idx)
            curr = vol.select(axis, idx + 1)
            stride = max(1, int(np.ceil(max(prev.shape) / 512)))
            prev = prev[::stride, ::stride]
            curr = curr[::stride, ::stride]
            counts = transition_counts(prev, curr, num_phases)
            rate = phase_change_rate(counts)
            if idx in band_set:
                band_rates[idx] = rate
                band_counts[idx] = counts
            elif idx in inner_set:
                inner_rates.append(rate)
                inner_counts.add_(counts)

        if not inner_rates or not band_idx:
            changes.append(None)
            tvs.append(None)
            deltas.append(None)
            continue
        inner_rate = float(torch.tensor(inner_rates).median())
        if inner_rate > 0.0:
            ratios = torch.tensor([band_rates[idx] for idx in band_idx]) / inner_rate
            changes.append(float(ratios[(ratios - 1.0).abs().argmax()]))
        else:
            changes.append(None)

        axis_tv = []
        axis_delta = []
        for idx in band_idx:
            seam_counts = band_counts[idx]
            axis_tv.append(transition_tv(seam_counts, inner_counts))
            axis_delta.append(continuation_delta(seam_counts, inner_counts))
        tvs.append(max(axis_tv))
        deltas.append(max(axis_delta))
    return SeamQuality(
        change_ratio=tuple(changes),
        transition_tv=tuple(tvs),
        continuation_delta=tuple(deltas),
    )


def show_slices(
    vol: torch.Tensor,
    num_phases: int,
) -> None:
    mid = tuple(size // 2 for size in vol.shape)
    imgs = (
        vol[mid[0], :, :],
        vol[:, mid[1], :],
        vol[:, :, mid[2]],
    )
    fig, panels = plt.subplots(1, 3, figsize=(11, 4))
    for axis, (panel, img) in enumerate(zip(panels, imgs, strict=True)):
        panel.imshow(
            img.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(f"axis {axis}")
        panel.axis("off")
    fig.suptitle("Joint tiled diffusion scale-up")
    fig.tight_layout()
    plt.show()


def show_unanchored_base_result(
    vol: torch.Tensor,
    base: torch.Tensor,
    center: torch.Tensor,
    num_phases: int,
) -> None:
    idx = base.shape[0] // 2
    base_plane = base[idx]
    center_plane = center[idx]
    diff_cmap = ListedColormap(("#f7f7f7", "#e63946"))
    fig, panels = plt.subplots(2, 3, figsize=(11, 7))

    for panel, (img, title) in zip(
        panels[0, :2],
        (
            (base_plane, "1. Base slice"),
            (center_plane, "2. Base area after scale"),
        ),
        strict=True,
    ):
        panel.imshow(
            img.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")

    panels[0, 2].imshow(
        (center_plane != base_plane).numpy(),
        cmap=diff_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    panels[0, 2].set_title("3. Scaled vs base")
    panels[0, 2].axis("off")

    start = tuple((size - base.shape[axis]) // 2 for axis, size in enumerate(vol.shape))
    mid = tuple(size // 2 for size in vol.shape)
    views = (
        vol[mid[0], :, :],
        vol[:, mid[1], :],
        vol[:, :, mid[2]],
    )
    for axis, (panel, img) in enumerate(zip(panels[1], views, strict=True)):
        plane_axes = tuple(value for value in range(3) if value != axis)
        top, left = (start[value] for value in plane_axes)
        height, width = (base.shape[value] for value in plane_axes)
        panel.imshow(
            img.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.add_patch(
            Rectangle(
                (left - 0.5, top - 0.5),
                width,
                height,
                fill=False,
                edgecolor="#00a6d6",
                linewidth=1.5,
            )
        )
        panel.set_title(f"{axis + 4}. Full volume axis {axis}")
        panel.axis("off")

    fig.suptitle(
        "Scale-up from unanchored base\n"
        f"cyan box: base {base.shape[0]}×{base.shape[1]}×{base.shape[2]} area"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def show_base_result(
    vol: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    center: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    num_phases: int,
) -> None:
    idx = select_display_index(base.shape[axis], indices)
    target_plane = target.movedim(axis, 0)[idx]
    base_plane = base.movedim(axis, 0)[idx]
    center_plane = center.movedim(axis, 0)[idx]
    start = tuple(
        (size - base.shape[value]) // 2 for value, size in enumerate(vol.shape)
    )
    full_plane = vol.movedim(axis, 0)[start[axis] + idx]
    diff_cmap = ListedColormap(("#f7f7f7", "#e63946"))
    fig, panels = plt.subplots(3, 3, figsize=(11, 10))
    plane_axes = tuple(value for value in range(3) if value != axis)
    full_height, full_width = (vol.shape[value] for value in plane_axes)
    top, left = (start[value] for value in plane_axes)

    planes = (
        (target_plane, "1. Reference anchor"),
        (center_plane, "2. Base area zoom"),
        (full_plane, "3. Full scaled slice"),
    )
    for number, (panel, (img, title)) in enumerate(zip(panels[0], planes, strict=True)):
        panel.imshow(
            img.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        if number == 2:
            panel.add_patch(
                Rectangle(
                    (left - 0.5, top - 0.5),
                    base_plane.shape[1],
                    base_plane.shape[0],
                    fill=False,
                    edgecolor="#00a6d6",
                    linewidth=1.5,
                )
            )
        panel.set_title(title)
        panel.axis("off")

    differences = (
        (base_plane != target_plane, "4. Base vs reference"),
        (center_plane != base_plane, "5. Scaled vs base"),
        (center_plane != target_plane, "6. Scaled vs reference"),
    )
    for number, (panel, (img, title)) in enumerate(
        zip(panels[1], differences, strict=True)
    ):
        extent = None
        if number == 2:
            extent = (
                left,
                left + img.shape[1],
                top + img.shape[0],
                top,
            )
        panel.imshow(
            img.numpy(),
            cmap=diff_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        if number == 2:
            panel.add_patch(
                Rectangle(
                    (left, top),
                    img.shape[1],
                    img.shape[0],
                    fill=False,
                    edgecolor="#00a6d6",
                    linewidth=1.5,
                )
            )
            panel.set_xlim(0, full_width)
            panel.set_ylim(full_height, 0)
            panel.set_facecolor("#e5e5e5")
            panel.set_xticks(())
            panel.set_yticks(())
            panel.set_title(title)
            continue
        panel.set_title(title)
        panel.axis("off")

    mid = tuple(size // 2 for size in vol.shape)
    views = (
        vol[mid[0], :, :],
        vol[:, mid[1], :],
        vol[:, :, mid[2]],
    )
    for view_axis, (panel, img) in enumerate(zip(panels[2], views, strict=True)):
        panel.imshow(
            img.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(f"{view_axis + 7}. Full volume axis {view_axis}")
        panel.axis("off")

    mode = f"{len(indices)} anchor planes" if indices else "unanchored base"
    fig.suptitle(
        f"Scale-up from {mode} · displayed slice {idx}\n"
        f"cyan box: base {target_plane.shape[0]}×{target_plane.shape[1]} area · "
        f"full slice: {full_height}×{full_width}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def show_napari(vol: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(vol.numpy(), name="scaled phases")
    viewer.dims.ndisplay = 3
    napari.run()


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


def format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


if __name__ == "__main__":
    main()
