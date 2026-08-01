"""Check evenly distributed soft anchor planes from a reference volume."""

import argparse
import sys
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
from src.train.weights import find_weights

VOLUME_PATH = PROJECT_ROOT / "data" / "generated" / "volumes" / "volume_000.tiff"
AXIS = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="number of evenly distributed anchor planes to use (default: 3)",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="show the complete generated phase volume in Napari",
    )
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be non-negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = (
        find_weights(PROJECT_ROOT / "run") if args.weights is None else args.weights
    )
    print("\nAnchor generation")
    print("-----------------")
    print(f"Weights : {Path(weights).resolve()}")
    print(f"Volume  : {VOLUME_PATH.resolve()}", flush=True)

    generator = load_generator(weights, device=device)
    if args.count > 0 and not generator.anchor_enabled:
        raise ValueError("selected weights were trained with anchors disabled.")

    if args.count > generator.patch_size:
        parser.error(f"--count must be at most {generator.patch_size}.")
    target = load_volume(
        VOLUME_PATH,
        patch_size=generator.patch_size,
        num_phases=generator.num_phases,
    )
    target_slices = get_slices(target, AXIS)
    indices = select_indices(target_slices.shape[0], args.count)
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
    print("Status   : generating...", flush=True)

    gen = generator.generate(anchors=anchors)
    print("Status   : complete", flush=True)
    gen_slices = get_slices(gen, AXIS)
    vol_acc = float((gen == target).to(torch.float32).mean())
    if indices:
        selected = torch.tensor(indices, dtype=torch.long)
        target_sel = target_slices.index_select(0, selected)
        gen_sel = gen_slices.index_select(0, selected)
        anchor_acc = float((gen_sel == target_sel).to(torch.float32).mean())
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
    slice_acc = float((~mismatch).to(torch.float32).mean())
    iou, recall = get_scores(
        score_gen,
        score_ref,
        num_phases=generator.num_phases,
    )

    print_quality(
        anchor_acc=anchor_acc,
        vol_acc=vol_acc,
        iou=iou,
        recall=recall,
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
) -> None:
    print("\nQuality")
    print("-------")
    score = "n/a" if anchor_acc is None else f"{anchor_acc:7.2%}"
    print(f"Selected planes : {score}")
    print(f"Complete volume : {vol_acc:7.2%}")
    print(f"Phase IoU       : {format_scores(iou)}")
    print(f"Phase recall    : {format_scores(recall)}")


def format_scores(values: list[float]) -> str:
    return "  ".join(
        f"phase {phase}: {value:.2%}" for phase, value in enumerate(values)
    )


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
        (gen, "2. Generated at anchor", False),
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

    mode = "Distributed soft anchors" if indices else "Unanchored baseline"
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


def get_scores(
    gen: torch.Tensor,
    target: torch.Tensor,
    num_phases: int,
) -> tuple[list[float], list[float]]:
    if gen.shape != target.shape:
        raise ValueError("generated and target volumes must have the same shape.")
    iou = []
    recall = []
    for phase in range(num_phases):
        pred = gen == phase
        expected = target == phase
        intersection = int((pred & expected).sum())
        union = int((pred | expected).sum())
        support = int(expected.sum())
        iou.append(1.0 if union == 0 else intersection / union)
        recall.append(1.0 if support == 0 else intersection / support)
    return iou, recall


def show_napari(vol: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(vol.numpy(), name="generated phases")
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
