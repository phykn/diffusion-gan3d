import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build import load_generator
from src.config import load_generation_settings


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
        help="show the complete 3D phase volume in Napari",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = args.weight
    print("\nGeneration")
    print("----------")
    print(f"Weight  : {weight.resolve()}", flush=True)

    generator = load_generator(weight, device=device)
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    shape = (generator.patch_size,) * 3
    print(f"Shape   : {' × '.join(map(str, shape))}")
    print(f"Device  : {device}")
    print("Conditioning : none")
    print(f"Margin  : {settings.margin} per outer face")
    print("Status  : generating...", flush=True)

    vol = generator.generate(
        vf=None,
        guidance=guidance,
        domain=args.domain,
        margin=settings.margin,
    )
    print("Status  : complete", flush=True)
    if args.out is not None:
        save_volume(vol, args.out)
    if args.napari:
        show_napari(vol)
    else:
        show_slices(vol, generator.num_phases)


def show_slices(
    vol: torch.Tensor,
    num_phases: int,
) -> None:
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


def save_volume(vol: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, vol.detach().cpu().to(torch.uint8).numpy())
    print(f"Output  : {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
