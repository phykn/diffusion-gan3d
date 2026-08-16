import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import (
    SliceSmoothness,
    measure_distance_divergence,
    measure_slice_smoothness,
    voxel_accuracy,
)
from src.volume import save_volume


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
        "--seed",
        type=int,
        default=0,
        help="reproducible diagnostic seed (default: 0)",
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="anchor plane axis (default: 0)",
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
        default=0.90,
        help="normalized anchor prediction strength from 0 to 1 (default: 0.90)",
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
        "--no-view",
        action="store_true",
        help="skip interactive visualization",
    )
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be non-negative.")
    anchor_count = args.count if args.anchor_strength > 0.0 else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = args.weight
    print(f"\nWeights : {Path(weight).resolve()}")

    generator = load_generator(weight, device=device)
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    if anchor_count > generator.patch_size:
        parser.error(f"--count must be at most {generator.patch_size}.")
    torch.manual_seed(args.seed)
    print("Generating reference...", flush=True)
    target = generator.generate(
        anchors=(),
        anchor_strength=0.0,
        guidance=guidance,
        domain=args.domain,
        margin=generator.default_margin,
    )
    target_slices = get_slices(target, args.axis)
    indices = select_indices(target_slices.shape[0], anchor_count)
    anchors = tuple(
        PlaneAnchor(image=target_slices[index], axis=args.axis, index=index)
        for index in indices
    )
    print_selection(
        shape=tuple(target.shape),
        device=device,
        axis=args.axis,
        indices=indices,
    )
    if indices:
        print(f"Guidance : strength {args.anchor_strength:.2f}")
    print("Generating conditioned sample...", flush=True)

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None

    gen = generator.generate(
        anchors=anchors,
        anchor_strength=args.anchor_strength,
        guidance=guidance,
        domain=args.domain,
        margin=generator.default_margin,
    )
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    print("Generating same-RNG baseline...", flush=True)
    baseline = generator.generate(
        anchors=(),
        anchor_strength=0.0,
        guidance=guidance,
        domain=args.domain,
        margin=generator.default_margin,
    )
    if args.out is not None:
        save_volume(gen, args.out)
        print(f"Saved   : {args.out.resolve()}", flush=True)
    gen_slices = get_slices(gen, args.axis)
    if indices:
        selected = torch.tensor(indices, dtype=torch.long)
        target_sel = target_slices.index_select(0, selected)
        gen_sel = gen_slices.index_select(0, selected)
        anchor_acc = voxel_accuracy(gen_sel, target_sel)
    else:
        anchor_acc = None
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

    print_quality(
        anchor_acc=anchor_acc,
        baseline_profile=baseline_profile,
        smoothness=smoothness,
    )
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
    print(f"Input   : {' x '.join(map(str, shape))}, {device}")
    print(f"Anchors : {len(indices)} planes on axis {axis}")


def print_quality(
    anchor_acc: float | None,
    baseline_profile: tuple[float | None, ...] = (),
    smoothness: SliceSmoothness | None = None,
) -> None:
    print("\nQuality")
    score = "n/a" if anchor_acc is None else f"{anchor_acc:7.2%}"
    print(f"Anchor match  : {score}")
    if smoothness is not None:
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
        print(f"Anchor effect  : {format_profile(baseline_profile)}")


def format_profile(
    values: tuple[float | None, ...],
) -> str:
    valid = [
        (distance, value) for distance, value in enumerate(values) if value is not None
    ]
    if not valid:
        return "n/a"
    anchor = next((value for distance, value in valid if distance == 0), None)
    mean = sum(value for _, value in valid) / len(valid)
    distance, farthest = valid[-1]
    return (
        f"anchor {format_percent(anchor)}, mean {format_percent(mean)}, "
        f"farthest {format_percent(farthest)} at distance {distance}"
    )


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


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


if __name__ == "__main__":
    main()
