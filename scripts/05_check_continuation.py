"""Check one-sided 3D continuation from a real boundary section."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import (
    measure_boundaries,
    measure_distance_divergence,
    measure_slice_smoothness,
    voxel_accuracy,
)
from src.generate import DEFAULT_ANCHOR_STRENGTH
from src.volume import save_volume

DISPLAY_DISTANCES = (0, 1, 2, 4, 8, 16, 32, 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument(
        "--anchor",
        type=Path,
        help="optional real 2D image; default uses an unconditioned sample",
    )
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--side", choices=("start", "end"), default="start")
    parser.add_argument(
        "--anchor-strength",
        type=unit_interval,
        default=DEFAULT_ANCHOR_STRENGTH,
    )
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--out", type=Path, help="optional generated TIFF path")
    parser.add_argument("--figure", type=Path, help="optional slice-strip PNG path")
    parser.add_argument(
        "--napari",
        action="store_true",
        help="inspect the generated volume and condition input in Napari",
    )
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(args.weight, device=device)
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    index = 0 if args.side == "start" else generator.patch_size - 1
    print(f"\nWeights : {args.weight.resolve()}")
    print(f"Boundary: axis {args.axis}, {args.side} plane {index}")
    print(f"Strength: {args.anchor_strength:.2f}")
    torch.manual_seed(args.seed)
    if args.anchor is None:
        print("Anchor  : unconditioned reference boundary")
        print("Generating unconditioned reference...", flush=True)
        reference = generator.generate(
            anchors=(),
            anchor_strength=0.0,
            guidance=guidance,
            domain=args.domain,
            margin=generator.default_margin,
        )
        anchor_image = reference.movedim(args.axis, 0)[index]
    else:
        anchor_image, crop = load_anchor_image(
            args.anchor,
            generator.patch_size,
            generator.num_phases,
        )
        print(f"Anchor  : {args.anchor.resolve()}, crop {crop}")
    anchor = PlaneAnchor(anchor_image, args.axis, index)

    print("Generating boundary-conditioned volume...", flush=True)

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    generated = generator.generate(
        anchors=(anchor,),
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
        save_volume(generated, args.out)
        print(f"Saved   : {args.out.resolve()}")
    print_quality(
        generated,
        baseline,
        anchor_image,
        args.axis,
        index,
        generator.num_phases,
    )
    if args.figure is not None or (not args.no_view and not args.napari):
        render_strip(
            generated,
            anchor_image,
            args.axis,
            args.side,
            generator.num_phases,
            output=args.figure,
            show=not args.no_view and not args.napari,
        )
    if args.napari and not args.no_view:
        show_napari(generated)


def load_anchor_image(
    path: Path,
    size: int,
    num_phases: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"anchor image was not found: {path}")
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("anchor image must be a 2D uint8 label image.")
    height, width = values.shape
    if height < size or width < size:
        raise ValueError(f"anchor image must contain a {size} x {size} crop.")
    top = (height - size) // 2
    left = (width - size) // 2
    crop = np.array(values[top : top + size, left : left + size], copy=True)
    if int(crop.max()) >= num_phases:
        raise ValueError(
            f"anchor image must contain phases from 0 to {num_phases - 1}."
        )
    return torch.from_numpy(crop).to(torch.long), (top, left, size, size)


def print_quality(
    generated: torch.Tensor,
    baseline: torch.Tensor,
    anchor: torch.Tensor,
    axis: int,
    index: int,
    num_phases: int,
) -> None:
    slices = generated.movedim(axis, 0)
    accuracy = voxel_accuracy(slices[index], anchor)
    boundary = measure_boundaries(generated, (index,), axis, num_phases)
    smoothness = measure_slice_smoothness(generated, (index,), axis, baseline)
    profile = measure_distance_divergence(
        generated,
        baseline,
        (index,),
        axis,
        generated.shape[axis] - 1,
    )
    print("\nQuality")
    print(f"Anchor match  : {accuracy:7.2%}")
    print(
        "First change  : "
        f"{format_percent(boundary.anchor_change)} vs ordinary "
        f"{format_percent(boundary.ordinary_change)} "
        f"({format_ratio(boundary.change_ratio)})"
    )
    print(
        "Smoothness    : "
        f"p95 {format_ratio(smoothness.p95_ratio)}, "
        f"peak {format_ratio(smoothness.max_ratio)}"
    )
    print(
        "Surface bumps : "
        f"{format_percent(smoothness.reversal_rate)} vs baseline "
        f"{format_percent(smoothness.baseline_reversal_rate)} "
        f"({format_ratio(smoothness.reversal_ratio)})"
    )
    print(f"Anchor effect : {format_profile(profile)}")


def render_strip(
    volume: torch.Tensor,
    anchor: torch.Tensor,
    axis: int,
    side: str,
    num_phases: int,
    *,
    output: Path | None,
    show: bool,
) -> None:
    slices = volume.movedim(axis, 0)
    if side == "end":
        slices = slices.flip(0)
    distances = tuple(
        distance
        for distance in (*DISPLAY_DISTANCES, len(slices) - 1)
        if distance < len(slices)
    )
    distances = tuple(dict.fromkeys(distances))
    images = (anchor, *(slices[distance] for distance in distances))
    index = 0 if side == "start" else len(slices) - 1
    titles = (
        f"Condition input (target)\naxis {axis}, index {index}",
        *(
            (
                f"Generated output\naxis {axis}, index {index}"
                if distance == 0
                else f"Generated d={distance}"
            )
            for distance in distances
        ),
    )
    columns = 5
    rows = (len(images) + columns - 1) // columns
    figure, panels = plt.subplots(
        rows, columns, squeeze=False, figsize=(12, 2.5 * rows)
    )
    cmap = plt.get_cmap("gray", num_phases)
    for panel, image, title in zip(panels.flat, images, titles, strict=False):
        panel.imshow(
            image.cpu().numpy(),
            cmap=cmap,
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title)
        panel.axis("off")
    input_panel = panels.flat[0]
    input_panel.add_patch(
        Rectangle(
            (-0.5, -0.5),
            anchor.shape[1],
            anchor.shape[0],
            fill=False,
            edgecolor="#F97316",
            linewidth=3.0,
        )
    )
    for panel in tuple(panels.flat)[len(images) :]:
        panel.axis("off")
    figure.suptitle(f"One-sided continuation from the {side} plane")
    figure.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Figure  : {output.resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def show_napari(volume: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(volume.cpu().numpy(), name="Generated output")
    viewer.dims.ndisplay = 3
    napari.run()


def format_profile(values: tuple[float | None, ...]) -> str:
    valid = [
        (distance, value) for distance, value in enumerate(values) if value is not None
    ]
    if not valid:
        return "n/a"
    near = valid[0][1]
    mean = sum(value for _, value in valid) / len(valid)
    distance, farthest = valid[-1]
    return (
        f"boundary {format_percent(near)}, mean {format_percent(mean)}, "
        f"farthest {format_percent(farthest)} at distance {distance}"
    )


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


if __name__ == "__main__":
    main()
