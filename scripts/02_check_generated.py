"""Generate and inspect one volume."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build import find_weights, load_generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--napari",
        action="store_true",
        help="show the complete 3D phase volume in Napari",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = find_weights(PROJECT_ROOT / "run")
    print("\nGeneration")
    print("----------")
    print(f"Weights : {weights.resolve()}", flush=True)

    generator = load_generator(weights, device=device)
    shape = (generator.patch_size,) * 3
    print(f"Shape   : {' × '.join(map(str, shape))}")
    print(f"Device  : {device}")
    print("Status  : generating...", flush=True)

    vol = generator.generate(vf=None)
    print("Status  : complete", flush=True)
    if args.napari:
        show_napari(vol)
    else:
        show_slices(vol, generator.num_phases)


def show_slices(vol: torch.Tensor, num_phases: int) -> None:
    mid = tuple(size // 2 for size in vol.shape)
    slices = (
        vol[mid[0], :, :],
        vol[:, mid[1], :],
        vol[:, :, mid[2]],
    )

    fig, panels = plt.subplots(1, 3, figsize=(10, 4))
    for axis, img in enumerate(slices):
        panels[axis].imshow(
            img,
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panels[axis].set_title(f"axis {axis}")
        panels[axis].axis("off")
    fig.suptitle("EMA model")
    fig.tight_layout()
    plt.show()


def show_napari(vol: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(vol.numpy(), name="generated phases")
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
