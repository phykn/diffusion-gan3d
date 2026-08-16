import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap, to_rgb
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOLUME_PATH = PROJECT_ROOT / "scripts" / "gt.tiff"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.png"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "paper"
PHASE_COLORS = ("#000000", "#9E9E9E")
CROP_SIZE = 128
ROI_COLOR = "#F97316"
ROI_POSITIONS = ((18, 16), (281, 58), (536, 24))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the core paper figures from a generated reference volume."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=VOLUME_PATH,
        help="3D uint8 label TIFF created by make_reference.py",
    )
    args = parser.parse_args()
    reference = args.reference.resolve()
    if not reference.is_file():
        raise FileNotFoundError(
            f"reference volume does not exist: {reference}. "
            "Run make_reference.py first or pass --reference."
        )

    volume = np.asarray(tifffile.imread(reference))
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError("reference must be a 3D uint8 phase volume.")

    colors = [to_rgb(color) for color in PHASE_COLORS]
    if int(volume.max()) >= len(colors):
        raise ValueError("paper palette does not define every volume phase.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_training_sample(
        SAMPLE_PATH,
        colors,
        OUTPUT_DIR / "01-training-data.png",
    )
    render_volume(volume, colors, OUTPUT_DIR / "02-generated-volume.png")
    render_slices(volume, colors, OUTPUT_DIR / "03-generated-slices.png")


def render_training_sample(
    path: Path,
    colors: list[tuple[float, float, float]],
    output: Path,
) -> None:
    with Image.open(path) as image:
        labels = np.asarray(image).copy()
    if labels.ndim != 2 or int(labels.max()) >= len(colors):
        raise ValueError(f"sample image contains an unsupported phase: {path}")
    palette = np.asarray(colors)
    rgb = np.rint(255 * palette[labels]).astype(np.uint8)
    rendered = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    for left, top in ROI_POSITIONS:
        right = left + CROP_SIZE - 1
        bottom = top + CROP_SIZE - 1
        if right >= rendered.width or bottom >= rendered.height:
            raise ValueError("paper crop ROI is outside the sample image.")
        draw.rectangle(
            (left, top, right, bottom),
            outline=ROI_COLOR,
            width=4,
        )
    rendered.save(output)
    print(f"Training: {output.resolve()}")


def render_volume(
    volume: np.ndarray,
    colors: list[tuple[float, float, float]],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(8, 7), facecolor="#f8fafc")
    panel = figure.add_subplot(111, projection="3d")
    draw_volume(panel, volume, colors)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print(f"Volume : {output.resolve()}")


def draw_volume(
    panel: plt.Axes,
    volume: np.ndarray,
    colors: list[tuple[float, float, float]],
) -> None:
    rgba = np.asarray([(*color, 1.0) for color in colors])
    last = tuple(size - 1 for size in volume.shape)
    center = tuple(size // 2 for size in volume.shape)

    # Complete outer faces away from the removed camera-facing octant.
    plot_face(
        panel,
        volume,
        rgba,
        axis=0,
        index=0,
        first=(0, last[1]),
        second=(0, last[2]),
        light=0.88,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=1,
        index=last[1],
        first=(0, last[0]),
        second=(0, last[2]),
        light=0.82,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=2,
        index=0,
        first=(0, last[0]),
        second=(0, last[1]),
        light=0.75,
    )

    # Camera-facing outer faces, split around the missing 1/8 corner.
    plot_face(
        panel,
        volume,
        rgba,
        axis=0,
        index=last[0],
        first=(0, last[1]),
        second=(0, center[2]),
        light=0.88,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=0,
        index=last[0],
        first=(center[1], last[1]),
        second=(center[2], last[2]),
        light=0.88,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=1,
        index=0,
        first=(0, last[0]),
        second=(0, center[2]),
        light=0.78,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=1,
        index=0,
        first=(0, center[0]),
        second=(center[2], last[2]),
        light=0.78,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=2,
        index=last[2],
        first=(0, center[0]),
        second=(0, last[1]),
        light=1.08,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=2,
        index=last[2],
        first=(center[0], last[0]),
        second=(center[1], last[1]),
        light=1.08,
    )

    # The three internal faces exposed by removing x-high/y-low/z-high.
    plot_face(
        panel,
        volume,
        rgba,
        axis=0,
        index=center[0],
        first=(0, center[1]),
        second=(center[2], last[2]),
        light=0.58,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=1,
        index=center[1],
        first=(center[0], last[0]),
        second=(center[2], last[2]),
        light=0.66,
    )
    plot_face(
        panel,
        volume,
        rgba,
        axis=2,
        index=center[2],
        first=(center[0], last[0]),
        second=(0, center[1]),
        light=0.68,
    )

    extent = np.asarray(last, dtype=float)
    draw_box(panel, extent)
    panel.set_xlim(0, extent[0])
    panel.set_ylim(0, extent[1])
    panel.set_zlim(0, extent[2])
    panel.set_box_aspect(extent)
    panel.view_init(elev=23, azim=-48)
    panel.set_proj_type("persp", focal_length=1.8)
    panel.set_axis_off()


def plot_face(
    panel: plt.Axes,
    volume: np.ndarray,
    rgba: np.ndarray,
    axis: int,
    index: int,
    first: tuple[int, int],
    second: tuple[int, int],
    light: float = 1.0,
) -> None:
    other_axes = tuple(value for value in range(3) if value != axis)
    first_cells = np.arange(first[0], first[1])
    second_cells = np.arange(second[0], second[1])
    first_coords = np.linspace(
        first[0],
        first[1],
        len(first_cells) + 1,
    )
    second_coords = np.linspace(
        second[0],
        second[1],
        len(second_cells) + 1,
    )
    first_grid, second_grid = np.meshgrid(
        first_coords,
        second_coords,
        indexing="ij",
    )
    coordinates = [None, None, None]
    coordinates[axis] = np.full_like(first_grid, index)
    coordinates[other_axes[0]] = first_grid
    coordinates[other_axes[1]] = second_grid
    phase_slice = np.take(volume, index, axis=axis)
    phases = phase_slice[np.ix_(first_cells, second_cells)]
    facecolors = rgba[phases].copy()
    if light < 1.0:
        facecolors[..., :3] *= light
    elif light > 1.0:
        facecolors[..., :3] = 1.0 - (1.0 - facecolors[..., :3]) / light
    panel.plot_surface(
        *coordinates,
        facecolors=facecolors,
        shade=False,
        linewidth=0,
        antialiased=False,
    )


def draw_box(panel: plt.Axes, size: np.ndarray) -> None:
    for axis in range(3):
        others = tuple(value for value in range(3) if value != axis)
        for first in (0.0, size[others[0]]):
            for second in (0.0, size[others[1]]):
                start = np.zeros(3)
                end = np.zeros(3)
                start[list(others)] = (first, second)
                end[list(others)] = (first, second)
                end[axis] = size[axis]
                panel.plot(
                    (start[0], end[0]),
                    (start[1], end[1]),
                    (start[2], end[2]),
                    color="#64748b",
                    linewidth=0.65,
                    alpha=0.32,
                )


def render_slices(
    volume: np.ndarray,
    colors: list[tuple[float, float, float]],
    output: Path,
) -> None:
    center = tuple(size // 2 for size in volume.shape)
    slices = (
        volume[center[0], :, :],
        volume[:, center[1], :],
        volume[:, :, center[2]],
    )
    cmap = ListedColormap(colors)
    figure, panels = plt.subplots(1, 3, figsize=(10.5, 3.5), facecolor="white")
    for axis, (panel, image) in enumerate(zip(panels, slices, strict=True)):
        panel.imshow(
            image,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(colors) - 0.5,
            interpolation="nearest",
        )
        panel.set_title(f"Axis {axis}", fontsize=12, pad=8)
        panel.axis("off")
    figure.tight_layout(pad=1.0, w_pad=1.2)
    figure.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    print(f"Slices : {output.resolve()}")


if __name__ == "__main__":
    main()
