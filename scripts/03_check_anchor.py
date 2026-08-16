import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import (
    continuation_delta,
    phase_change_rate,
    phase_iou,
    phase_recall,
    transition_counts,
    transition_tv,
    voxel_accuracy,
)

AXIS = 0
DISTANCE_PROFILE_RADIUS = 24


@dataclass(frozen=True)
class BoundaryQuality:
    anchor_change: float | None
    ordinary_change: float | None
    change_ratio: float | None
    transition_tv: float | None
    continuation_delta: float | None


def main() -> None:
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
        required=True,
        help="ground-truth TIFF volume used to build anchors",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="number of evenly distributed anchor planes to use (default: 3)",
    )
    parser.add_argument(
        "--anchor-strength",
        type=unit_interval,
        default=1.0,
        help="normalized anchor prediction strength from 0 to 1 (default: 1)",
    )
    parser.add_argument(
        "--anchor-sigma",
        type=positive_float,
        default=2.0,
        help="Gaussian anchor influence radius in voxels (default: 2)",
    )
    parser.add_argument(
        "--guidance",
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
        help="show the complete generated phase volume in Napari",
    )
    parser.add_argument(
        "--compare-unconditioned",
        action="store_true",
        help="also generate the same-RNG unconditioned baseline",
    )
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be non-negative.")
    anchor_count = args.count if args.anchor_strength > 0.0 else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = args.weight
    print("\nAnchor generation")
    print("-----------------")
    print(f"Weight  : {Path(weight).resolve()}")
    print(f"GT      : {args.gt.resolve()}", flush=True)

    generator = load_generator(weight, device=device)
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    if anchor_count > generator.patch_size:
        parser.error(f"--count must be at most {generator.patch_size}.")
    target = load_volume(
        args.gt,
        patch_size=generator.patch_size,
        num_phases=generator.num_phases,
    )
    target_slices = get_slices(target, AXIS)
    indices = select_indices(target_slices.shape[0], anchor_count)
    anchors = tuple(
        PlaneAnchor(image=target_slices[index], axis=AXIS, index=index)
        for index in indices
    )
    print_selection(
        shape=tuple(target.shape),
        device=device,
        axis=AXIS,
        indices=indices,
    )
    mode = "learned conditional anchors" if indices else "none"
    print(f"Conditioning : {mode}")
    if indices:
        print(f"Anchor strength : {args.anchor_strength:.2f}")
        print(f"Anchor sigma    : {args.anchor_sigma:g} voxels")
    print(f"Margin  : {settings.margin} per outer face")
    print("Status   : generating...", flush=True)

    cpu_rng = torch.random.get_rng_state() if args.compare_unconditioned else None
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if args.compare_unconditioned and device.type == "cuda"
        else None
    )

    gen = generator.generate(
        anchors=anchors,
        anchor_strength=args.anchor_strength,
        anchor_sigma=args.anchor_sigma,
        guidance=guidance,
        domain=args.domain,
        margin=settings.margin,
    )
    print("Status   : complete", flush=True)
    baseline = None
    if args.compare_unconditioned and indices:
        assert cpu_rng is not None
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        print("Baseline : generating same-RNG unconditioned volume...", flush=True)
        baseline = generator.generate(
            anchor_strength=0.0,
            guidance=guidance,
            domain=args.domain,
            margin=settings.margin,
        )
        print("Baseline : complete", flush=True)
    if args.out is not None:
        save_volume(gen, args.out)
    gen_slices = get_slices(gen, AXIS)
    vol_acc = voxel_accuracy(gen, target)
    if indices:
        selected = torch.tensor(indices, dtype=torch.long)
        target_sel = target_slices.index_select(0, selected)
        gen_sel = gen_slices.index_select(0, selected)
        anchor_acc = voxel_accuracy(gen_sel, target_sel)
        score_gen = gen_sel
        score_ref = target_sel
    else:
        anchor_acc = None
        score_gen = gen
        score_ref = target
    index = select_display_index(generator.patch_size, indices)
    target_plane = target_slices[index]
    gen_plane = gen_slices[index]
    mismatch = gen_plane != target_plane
    slice_acc = voxel_accuracy(gen_plane, target_plane)
    iou = phase_iou(score_gen, score_ref, generator.num_phases)
    recall = phase_recall(score_gen, score_ref, generator.num_phases)
    boundary = measure_boundaries(
        gen,
        indices,
        AXIS,
        generator.num_phases,
    )
    distance_profile = measure_distance_changes(
        gen,
        indices,
        AXIS,
        DISTANCE_PROFILE_RADIUS,
    )
    baseline_profile = (
        ()
        if baseline is None
        else measure_distance_divergence(
            gen,
            baseline,
            indices,
            AXIS,
            DISTANCE_PROFILE_RADIUS,
        )
    )

    print_quality(
        anchor_acc=anchor_acc,
        vol_acc=vol_acc,
        iou=iou,
        recall=recall,
        boundary=boundary,
        distance_profile=distance_profile,
        baseline_profile=baseline_profile,
    )
    if args.napari:
        show_napari(gen)
    else:
        show_result(
            gen,
            target_plane,
            gen_plane,
            mismatch,
            indices=indices,
            axis=AXIS,
            index=index,
            num_phases=generator.num_phases,
            accuracy=slice_acc,
        )


def select_indices(size: int, count: int) -> tuple[int, ...]:
    if size < 1:
        raise ValueError("volume depth must be positive.")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= size:
        raise ValueError("count must be between 0 and the volume depth.")
    if count == 0:
        return ()
    return tuple(((2 * index + 1) * size) // (2 * count) for index in range(count))


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def select_display_index(size: int, indices: tuple[int, ...]) -> int:
    center = size // 2
    if not indices:
        return center
    return min(indices, key=lambda index: (abs(index - center), index))


def print_selection(
    shape: tuple[int, ...],
    device: torch.device,
    axis: int,
    indices: tuple[int, ...],
) -> None:
    depth = shape[axis]
    coverage = len(indices) / depth
    print(f"Shape    : {' × '.join(map(str, shape))}")
    print(f"Device   : {device}")
    print(f"Anchors  : {len(indices)} planes")
    print(f"Coverage : {coverage:.2%} ({len(indices)} / {depth} slices)")
    print(f"Axis     : {axis}")


def print_quality(
    anchor_acc: float | None,
    vol_acc: float,
    iou: list[float],
    recall: list[float],
    boundary: BoundaryQuality,
    distance_profile: tuple[float | None, ...],
    baseline_profile: tuple[float | None, ...] = (),
) -> None:
    print("\nQuality")
    print("-------")
    score = "n/a" if anchor_acc is None else f"{anchor_acc:7.2%}"
    print(f"Anchor match      : {score}")
    print(f"Complete volume : {vol_acc:7.2%}")
    print(f"Phase IoU       : {format_scores(iou)}")
    print(f"Phase recall    : {format_scores(recall)}")
    print("\nAnchor boundary")
    print("---------------")
    print(f"Anchor sides       : {format_score(boundary.anchor_change)}")
    print(f"Ordinary planes    : {format_score(boundary.ordinary_change)}")
    print(f"Change ratio       : {format_value(boundary.change_ratio)}")
    print(f"Transition TV      : {format_value(boundary.transition_tv)}")
    print(f"Continuation delta : {format_value(boundary.continuation_delta)}")
    if distance_profile:
        print("\nDistance profile")
        print("----------------")
        for distance, change in enumerate(distance_profile):
            print(f"Distance {distance:2d} : {format_score(change)}")
    if baseline_profile:
        print("\nSame-RNG baseline divergence")
        print("----------------------------")
        for distance, change in enumerate(baseline_profile):
            print(f"Distance {distance:2d} : {format_score(change)}")


def format_scores(values: tuple[float, ...]) -> str:
    return "  ".join(
        f"phase {phase}: {value:.2%}" for phase, value in enumerate(values)
    )


def format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def measure_boundaries(
    vol: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    num_phases: int,
) -> BoundaryQuality:
    slices = get_slices(vol, axis)
    pair_count = slices.shape[0] - 1
    boundary_indices = sorted(
        {
            pair
            for index in indices
            for pair in (index - 1, index)
            if 0 <= pair < pair_count
        }
    )
    boundary_set = set(boundary_indices)
    ordinary_indices = [pair for pair in range(pair_count) if pair not in boundary_set]
    if not boundary_indices or not ordinary_indices:
        return BoundaryQuality(None, None, None, None, None)

    boundary_counts = transition_counts(
        slices[boundary_indices],
        slices[[index + 1 for index in boundary_indices]],
        num_phases,
    )
    ordinary_counts = transition_counts(
        slices[ordinary_indices],
        slices[[index + 1 for index in ordinary_indices]],
        num_phases,
    )
    boundary_change = phase_change_rate(boundary_counts)
    ordinary_change = phase_change_rate(ordinary_counts)
    ratio = None if ordinary_change == 0.0 else boundary_change / ordinary_change
    tv_value = transition_tv(boundary_counts, ordinary_counts)
    continuation_value = continuation_delta(boundary_counts, ordinary_counts)
    return BoundaryQuality(
        boundary_change,
        ordinary_change,
        ratio,
        tv_value,
        continuation_value,
    )


def measure_distance_changes(
    vol: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    max_distance: int,
) -> tuple[float | None, ...]:
    if not indices:
        return ()
    slices = get_slices(vol, axis)
    buckets: list[list[int]] = [[] for _ in range(max_distance + 1)]
    for pair in range(slices.shape[0] - 1):
        distance = min(
            min(abs(pair - index), abs(pair + 1 - index)) for index in indices
        )
        if distance <= max_distance:
            buckets[distance].append(pair)
    profile = []
    for pairs in buckets:
        if not pairs:
            profile.append(None)
            continue
        left = slices[pairs]
        right = slices[[pair + 1 for pair in pairs]]
        profile.append(float((left != right).to(torch.float32).mean()))
    return tuple(profile)


def measure_distance_divergence(
    anchored: torch.Tensor,
    baseline: torch.Tensor,
    indices: tuple[int, ...],
    axis: int,
    max_distance: int,
) -> tuple[float | None, ...]:
    if not indices:
        return ()
    anchored_slices = get_slices(anchored, axis)
    baseline_slices = get_slices(baseline, axis)
    changes = (anchored_slices != baseline_slices).to(torch.float32).mean((1, 2))
    positions = torch.arange(anchored_slices.shape[0], dtype=torch.long)
    anchors = torch.tensor(indices, dtype=torch.long)
    distances = (positions[:, None] - anchors[None, :]).abs().amin(dim=1)
    profile = []
    for distance in range(max_distance + 1):
        selected = changes[distances == distance]
        profile.append(None if selected.numel() == 0 else float(selected.mean()))
    return tuple(profile)


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
    slices = get_slices(vol, axis)
    cmap = "gray"
    diff_cmap = ListedColormap(("#f7f7f7", "#e63946"))
    fig, panels = plt.subplots(3, 3, figsize=(10, 9))

    comparisons = (
        (target, "1. Input anchor" if indices else "1. Reference center", False),
        (gen, "2. Bridge-conditioned output", False),
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
        section = get_slices(vol, view_axis)[vol.shape[view_axis] // 2]
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


def load_volume(
    path: Path,
    patch_size: int,
    num_phases: int,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"anchor volume was not found: {path}")
    vol = np.asarray(tifffile.imread(path))
    if vol.ndim != 3 or vol.size == 0:
        raise ValueError("anchor volume must be a non-empty 3D array.")
    if vol.dtype != np.uint8:
        raise ValueError(
            f"anchor volume must contain uint8 phases, got {vol.dtype}: {path}"
        )
    if int(vol.max()) >= num_phases:
        raise ValueError(
            f"anchor volume must contain phases from 0 to {num_phases - 1}."
        )

    tensor = torch.from_numpy(np.array(vol, copy=True)).long()
    if vol.shape != (patch_size, patch_size, patch_size):
        tensor = F.interpolate(
            tensor[None, None].to(torch.float32),
            size=(patch_size, patch_size, patch_size),
            mode="nearest",
        )[0, 0].to(torch.long)
    return tensor


def get_slices(vol: torch.Tensor, axis: int) -> torch.Tensor:
    if vol.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return vol.movedim(axis, 0)


def show_napari(vol: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(vol.numpy(), name="generated phases")
    viewer.dims.ndisplay = 3
    napari.run()


def save_volume(vol: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, vol.detach().cpu().to(torch.uint8).numpy())
    print(f"Output   : {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
