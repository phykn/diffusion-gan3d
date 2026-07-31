"""Check all soft anchor planes from the fixed reference volume."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._anchor_check import (
    load_volume,
    score_phases,
    slice_axis,
)
from src.anchor import PlaneAnchor
from src.build import load_sampler
from src.generate.sample import find_weights

VOLUME_PATH = PROJECT_ROOT / "data" / "generated" / "volumes" / "volume_000.tiff"
AXIS = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = (
        find_weights(PROJECT_ROOT / "run") if args.weights is None else args.weights
    )
    sampler = load_sampler(weights, device=device)
    if not sampler.anchor_enabled:
        raise ValueError("selected weights were trained with anchors disabled.")

    target = load_volume(
        VOLUME_PATH,
        patch_size=sampler.patch_size,
        num_phases=sampler.num_phases,
    )
    target_slices = slice_axis(target, AXIS)
    anchors = tuple(
        PlaneAnchor(
            labels=target_slices[index],
            axis=AXIS,
            index=index,
        )
        for index in range(target_slices.shape[0])
    )
    generated = sampler.generate(
        anchors=anchors,
        enforce=False,
    )
    generated_slices = slice_axis(generated, AXIS)
    slice_accuracy = (
        (generated_slices == target_slices).to(torch.float32).mean(dim=(1, 2))
    )
    overall_accuracy = float((generated == target).to(torch.float32).mean())
    worst_index = int(slice_accuracy.argmin())
    mismatch = generated_slices[worst_index] != target_slices[worst_index]
    iou, recall = score_phases(
        generated,
        target,
        num_phases=sampler.num_phases,
    )

    _show(
        target_slices,
        generated_slices,
        mismatch,
        slice_accuracy,
        worst_index=worst_index,
        overall_accuracy=overall_accuracy,
        num_phases=sampler.num_phases,
    )
    print(f"weights={Path(weights).resolve()}")
    print(f"volume={VOLUME_PATH.resolve()}")
    print(f"prepared_shape={tuple(target.shape)}")
    print(f"anchor_axis={AXIS} anchor_count={len(anchors)}")
    print(f"accuracy={overall_accuracy:.4f}")
    print(f"worst_slice={worst_index}")
    print(f"worst_accuracy={float(slice_accuracy[worst_index]):.4f}")
    print(f"phase_iou={[round(value, 4) for value in iou]}")
    print(f"phase_recall={[round(value, 4) for value in recall]}")


def _show(
    target: torch.Tensor,
    generated: torch.Tensor,
    mismatch: torch.Tensor,
    slice_accuracy: torch.Tensor,
    *,
    worst_index: int,
    overall_accuracy: float,
    num_phases: int,
) -> None:
    difference_cmap = ListedColormap(("#f7f7f7", "#e63946"))
    figure = plt.figure(figsize=(10, 7))
    grid = figure.add_gridspec(2, 3, height_ratios=(1, 0.8))
    comparisons = (
        (target[worst_index], "Target", False),
        (generated[worst_index], "Generated", False),
        (mismatch, "Difference (red = mismatch)", True),
    )
    for column, (image, title, difference) in enumerate(comparisons):
        panel = figure.add_subplot(grid[0, column])
        panel.imshow(
            image.numpy(),
            cmap=difference_cmap if difference else "gray",
            vmin=0 if difference else -0.5,
            vmax=1 if difference else num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(f"{title}\nslice {worst_index}")
        panel.axis("off")

    panel = figure.add_subplot(grid[1, :])
    values = slice_accuracy.numpy()
    indices = np.arange(values.shape[0])
    panel.plot(indices, values, linewidth=2, label="Per-slice accuracy")
    panel.axhline(
        overall_accuracy,
        color="#e63946",
        linestyle="--",
        label=f"Overall {100 * overall_accuracy:.1f}%",
    )
    panel.scatter(
        [worst_index],
        [values[worst_index]],
        color="#e63946",
        zorder=3,
        label=f"Worst {100 * values[worst_index]:.1f}%",
    )
    panel.set(
        xlabel=f"Slice index (axis {AXIS})",
        ylabel="Accuracy",
        xlim=(0, values.shape[0] - 1),
        ylim=(0, 1),
    )
    panel.grid(alpha=0.25)
    panel.legend()

    figure.suptitle(
        f"All-slice soft anchors · {100 * overall_accuracy:.1f}% matched\n"
        f"{values.shape[0]} planes · axis {AXIS}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


if __name__ == "__main__":
    main()
