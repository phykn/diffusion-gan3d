import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build import load_sampler
from src.generate.sample import find_weights

FRACTION = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--napari",
        action="store_true",
        help="show the complete 3D label volume in Napari",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = find_weights(PROJECT_ROOT / "run")
    sampler = load_sampler(weights, device=device)
    labels = sampler.generate(fraction=FRACTION)
    if args.napari:
        _show_napari(labels)
    else:
        _show_slices(labels, sampler.num_phases)
    print(f"weights={Path(weights).resolve()}")
    print(f"shape={tuple(labels.shape)}")


def _show_slices(labels: torch.Tensor, num_phases: int) -> None:
    middle = tuple(size // 2 for size in labels.shape)
    slices = (
        labels[middle[0], :, :],
        labels[:, middle[1], :],
        labels[:, :, middle[2]],
    )

    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    for axis, image in enumerate(slices):
        axes[axis].imshow(
            image,
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        axes[axis].set_title(f"axis {axis}")
        axes[axis].axis("off")
    figure.suptitle("EMA model")
    figure.tight_layout()
    plt.show()


def _show_napari(labels: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(labels.numpy(), name="generated phases")
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
