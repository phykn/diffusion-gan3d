"""Check one learned soft anchor plane from the fixed reference volume."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._anchor_check import (
    load_volume,
    score_phases,
    slice_axis,
)
from src.generate import (
    PlaneAnchor,
    Sampler,
    find_weights,
    load_model,
)

VOLUME_PATH = PROJECT_ROOT / "data" / "generated" / "volumes" / "volume_000.tif"
AXIS = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = (
        find_weights(PROJECT_ROOT / "run") if args.weights is None else args.weights
    )
    model, cfg = load_model(weights, device=device)
    if not cfg.anchor.enabled:
        raise ValueError("selected weights were trained with anchors disabled.")

    target, starts = load_volume(
        VOLUME_PATH,
        crop_size=cfg.data.crop_size,
        output_size=cfg.data.patch_size,
        num_phases=cfg.data.num_phases,
    )
    index = cfg.data.patch_size // 2
    target_plane = slice_axis(target, AXIS)[index]
    generated = Sampler(
        model,
        cfg,
        device=device,
    ).generate(
        anchors=(
            PlaneAnchor(
                labels=target_plane,
                axis=AXIS,
                index=index,
            ),
        ),
    )
    generated_slices = slice_axis(generated, AXIS)
    generated_plane = generated_slices[index]
    mismatch = generated_plane != target_plane
    accuracy = float((~mismatch).to(torch.float32).mean())
    iou, recall = score_phases(
        generated_plane,
        target_plane,
        num_phases=cfg.data.num_phases,
    )

    _show(
        generated,
        target_plane,
        generated_plane,
        mismatch,
        axis=AXIS,
        index=index,
        num_phases=cfg.data.num_phases,
        accuracy=accuracy,
    )
    print(f"weights={Path(weights).resolve()}")
    print(f"volume={VOLUME_PATH.resolve()}")
    print(f"crop_start={starts}")
    print(f"prepared_shape={tuple(target.shape)}")
    print(f"anchor_axis={AXIS} anchor_index={index}")
    print(f"accuracy={accuracy:.4f}")
    print(f"phase_iou={[round(value, 4) for value in iou]}")
    print(f"phase_recall={[round(value, 4) for value in recall]}")


def _show(
    volume: torch.Tensor,
    target: torch.Tensor,
    generated: torch.Tensor,
    mismatch: torch.Tensor,
    *,
    axis: int,
    index: int,
    num_phases: int,
    accuracy: float,
) -> None:
    slices = slice_axis(volume, axis)
    phase_cmap = "gray"
    difference_cmap = ListedColormap(("#f7f7f7", "#e63946"))
    figure, panels = plt.subplots(3, 3, figsize=(10, 9))

    comparisons = (
        (target, "1. Input anchor", False),
        (generated, "2. Generated at anchor", False),
        (mismatch, "3. Difference (red = mismatch)", True),
    )
    for panel, (image, title, difference) in zip(
        panels[0],
        comparisons,
        strict=True,
    ):
        panel.imshow(
            image.numpy(),
            cmap=difference_cmap if difference else phase_cmap,
            vmin=0 if difference else -0.5,
            vmax=1 if difference else num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")

    neighbors = (
        (index - 1, f"4. Before (slice {index - 1})"),
        (index, f"5. Anchor position (slice {index})"),
        (index + 1, f"6. After (slice {index + 1})"),
    )
    for panel, (slice_index, title) in zip(
        panels[1],
        neighbors,
        strict=True,
    ):
        panel.imshow(
            slices[slice_index].numpy(),
            cmap=phase_cmap,
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")

    orthogonal_axes = tuple(value for value in (0, 1, 2) if value != axis)
    for panel, image_axis in zip(
        panels[2, :2],
        orthogonal_axes,
        strict=True,
    ):
        section = slice_axis(volume, image_axis)[volume.shape[image_axis] // 2]
        anchor_dimension = axis if axis < image_axis else axis - 1
        oriented = section if anchor_dimension == 0 else section.T
        panel.imshow(
            oriented.numpy(),
            cmap=phase_cmap,
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.axhline(index, color="#e63946", linewidth=1.5)
        panel.set_title(f"Orthogonal view from axis {image_axis}")
        panel.axis("off")

    explanation = panels[2, 2]
    explanation.axis("off")
    explanation.text(
        0.0,
        0.9,
        "Soft-anchor check",
        fontsize=11,
        fontweight="bold",
        transform=explanation.transAxes,
    )
    explanation.text(
        0.0,
        0.72,
        "The input plane conditions every\n"
        "reverse step. It is not copied into\n"
        "the final volume.\n\n"
        "The red lines show where the anchor\n"
        "crosses the orthogonal sections.",
        fontsize=10,
        linespacing=1.4,
        va="top",
        transform=explanation.transAxes,
    )

    figure.suptitle(
        f"Single soft anchor · {100 * accuracy:.1f}% matched\n"
        f"axis {axis} · slice {index}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


if __name__ == "__main__":
    main()
