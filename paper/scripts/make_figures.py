"""Render the paper figures from evaluate_lr.py results and its cached volumes."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, to_rgb
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.evaluate_lr import (
    CACHE,
    DISTANCES,
    EXTERNAL_CROP,
    OUTPUT,
    real_crops,
)
from scripts.common.metrics import read_metrics
from src.build.data import build_datasets
from src.config.train import load_train_config
from src.evaluate.structure import label_sections

OUTPUT_DIR = OUTPUT.parent
PHASE_RGB = [to_rgb(color) for color in ("#000000", "#9E9E9E")]
ROI_COLOR = "#F97316"
# (left, top) of the example crops drawn on the training image.
ROI_POSITIONS = ((18, 16), EXTERNAL_CROP, (536, 24))
# Categorical slots 1-3 of the validated default palette, in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MUTED = "#6b7280"
INK = "#1f2937"
CMAP = ListedColormap(PHASE_RGB)


def main() -> None:
    results = json.loads(OUTPUT.read_text())
    run = PROJECT_ROOT / results["run"]
    best = results["best_step"]
    cfg = load_train_config(run / "train.yaml")
    dataset = build_datasets(cfg)[0][0]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    volumes = torch.load(CACHE / f"step{best}.pt", weights_only=True)
    render_training_image(dataset)
    render_volume(volumes[0].numpy())
    render_sections(volumes[0], dataset)
    render_training(results, run, cfg["burn_in"]["steps"])
    render_anchor(results)


def render_training_image(dataset) -> None:
    labels = dataset.decode(dataset.path_groups[0][0].resolve())
    rgb = np.rint(255 * np.asarray(PHASE_RGB)[labels]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    size = dataset.crop_size
    for left, top in ROI_POSITIONS:
        draw.rectangle(
            (left, top, left + size - 1, top + size - 1), outline=ROI_COLOR, width=4
        )
    path = OUTPUT_DIR / "01-training-data.png"
    image.save(path)
    print(f"Figure: {path.relative_to(PROJECT_ROOT)}")


def render_volume(volume: np.ndarray) -> None:
    figure = plt.figure(figsize=(8, 7), facecolor="#f8fafc")
    panel = figure.add_subplot(111, projection="3d")
    draw_volume(panel, volume, PHASE_RGB)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(figure, "02-generated-volume.png", pad=0.03)


def draw_volume(
    panel: plt.Axes,
    volume: np.ndarray,
    colors: list[tuple[float, float, float]],
) -> None:
    rgba = np.asarray([(*color, 1.0) for color in colors])
    last = tuple(size - 1 for size in volume.shape)
    center = tuple(size // 2 for size in volume.shape)

    faces = (
        # (axis, index, first, second, light); first the complete outer faces
        # away from the removed camera-facing octant,
        (0, 0, (0, last[1]), (0, last[2]), 0.88),
        (1, last[1], (0, last[0]), (0, last[2]), 0.82),
        (2, 0, (0, last[0]), (0, last[1]), 0.75),
        # then the camera-facing outer faces split around the missing 1/8 corner,
        (0, last[0], (0, last[1]), (0, center[2]), 0.88),
        (0, last[0], (center[1], last[1]), (center[2], last[2]), 0.88),
        (1, 0, (0, last[0]), (0, center[2]), 0.78),
        (1, 0, (0, center[0]), (center[2], last[2]), 0.78),
        (2, last[2], (0, center[0]), (0, last[1]), 1.08),
        (2, last[2], (center[0], last[0]), (center[1], last[1]), 1.08),
        # and the three internal faces exposed by removing x-high/y-low/z-high.
        (0, center[0], (0, center[1]), (center[2], last[2]), 0.58),
        (1, center[1], (center[0], last[0]), (center[2], last[2]), 0.66),
        (2, center[2], (center[0], last[0]), (0, center[1]), 0.68),
    )
    for face in faces:
        plot_face(panel, volume, rgba, *face)

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
    light: float,
) -> None:
    other_axes = tuple(value for value in range(3) if value != axis)
    first_cells = np.arange(first[0], first[1])
    second_cells = np.arange(second[0], second[1])
    first_grid, second_grid = np.meshgrid(
        np.arange(first[0], first[1] + 1, dtype=float),
        np.arange(second[0], second[1] + 1, dtype=float),
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
        rstride=1,
        cstride=1,
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


def show(panel, image, title=None):
    panel.imshow(
        np.asarray(image), cmap=CMAP, vmin=-0.5, vmax=1.5, interpolation="nearest"
    )
    if title:
        panel.set_title(title, fontsize=10, color=INK, pad=5)
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_visible(False)


def render_sections(volume: torch.Tensor, dataset) -> None:
    labels = label_sections(real_crops(dataset)).argmax(1)
    center = volume.shape[0] // 2
    figure, panels = plt.subplots(2, 3, figsize=(8.4, 5.8))
    for column, index in enumerate((0, 1, 2)):
        show(panels[0, column], labels[index], f"Real crop {index + 1}")
    for column, (name, section) in enumerate(
        zip(
            ("xy", "xz", "yz"),
            (volume[center], volume[:, center], volume[:, :, center]),
        )
    ):
        show(panels[1, column], section, f"Generated {name}")
    figure.tight_layout(pad=0.6, h_pad=1.0)
    save(figure, "03-generated-slices.png")


def render_training(results, run: Path, burn_in: int) -> None:
    steps = results["selection"]
    x = [row["step"] for row in steps]
    best = results["best_step"]
    real = results["real"]["half_split_errors"]
    rows = read_metrics(run / "tensorboard")
    figure, panels = plt.subplots(1, 3, figsize=(12, 3.5))

    panel = panels[0]
    panel.plot(x, [row["structure"] for row in steps], color=SERIES[0], lw=2)
    mark_schedule(panel, best, burn_in)
    panel.set_title("(a) Structure score (selection)", loc="left", color=INK)
    panel.set_xlabel("Step")
    panel.set_ylabel("Mean error / real split error")

    panel = panels[1]
    names = (
        ("two_point_mae", "Two-point"),
        ("chord_tv", "Chord"),
        ("cluster_error", "Cluster"),
    )
    for color, (key, label) in zip(SERIES, names):
        values = np.asarray([row[key] for row in steps]) / max(real[key], 1e-6)
        panel.plot(x, values, color=color, lw=2, label=label)
    mark_schedule(panel, best, burn_in)
    panel.legend(frameon=False, fontsize=9, labelcolor=INK)
    panel.set_title("(b) Components / real split error", loc="left", color=INK)
    panel.set_xlabel("Step")

    panel = panels[2]
    for color, (kind, label) in zip(SERIES, (("real", "Real"), ("fake", "Generated"))):
        tags = [f"score/xy_xz_yz/{axis}/{kind}" for axis in range(3)]
        points = [
            (row["step"], np.mean([row[t] for t in tags]))
            for row in rows
            if all(row.get(t) is not None for t in tags)
        ]
        s, v = np.asarray(points).T
        smooth = np.convolve(v, np.ones(200) / 200, mode="valid")
        panel.plot(s[len(s) - len(smooth) :], smooth, color=color, lw=2)
        panel.annotate(
            label,
            (s[-1], smooth[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=INK,
            va="center",
            fontsize=9,
        )
    mark_schedule(panel, best, burn_in)
    panel.set_title("(c) Mean critic logit", loc="left", color=INK)
    panel.set_xlabel("Step")
    panel.set_xlim(0, x[-1] * 1.22)
    panel.set_xticks(range(0, x[-1] + 1, 2000))
    figure.tight_layout(w_pad=1.5)
    save(figure, "lr-training.png")


def mark_schedule(panel, best, burn_in):
    panel.axvspan(0, burn_in, color="#e5e7eb", alpha=0.6, lw=0)
    panel.axvline(best, color=MUTED, lw=1, ls="--")
    panel.grid(axis="y", color="#e5e7eb", lw=0.8)


def render_anchor(results) -> None:
    anchor = results["anchor"]
    external = torch.load(CACHE / "anchor-external.pt", weights_only=True)
    plane = external["plane"]
    row = next(r for r in external["rows"] if r["seed"] == 0 and r["index"] != 0)
    index = row["index"]
    figure = plt.figure(figsize=(12, 5.6))
    grid = figure.add_gridspec(2, 6, height_ratios=(0.62, 1), hspace=0.3, wspace=0.45)
    show(figure.add_subplot(grid[0, 0]), plane, "(a) Supplied section")
    show(figure.add_subplot(grid[0, 1]), row["anchored"][index], "(b) DPS, d = 0")
    show(
        figure.add_subplot(grid[0, 2]),
        row["baseline"][index],
        "(c) Same noise, no anchor",
    )
    for column, d in zip((3, 4, 5), (2, 8, 24)):
        show(
            figure.add_subplot(grid[0, column]),
            row["anchored"][index + d],
            f"DPS, d = {d}",
        )

    panel = figure.add_subplot(grid[1, 0:3])
    internal = anchor["internal"]
    distances = [
        d for d in DISTANCES if all(str(d) in r["reference"] for r in internal)
    ]
    for color, (key, label) in zip(
        SERIES, (("anchored", "DPS"), ("baseline", "Same-noise baseline"))
    ):
        values = np.asarray(
            [[r["reference"][str(d)][key] for d in distances] for r in internal]
        )
        mean, std = values.mean(0), values.std(0, ddof=1)
        panel.plot(
            distances, 100 * mean, color=color, lw=2, marker="o", ms=4, label=label
        )
        panel.fill_between(
            distances,
            100 * (mean - std),
            100 * (mean + std),
            color=color,
            alpha=0.15,
            lw=0,
        )
    panel.legend(frameon=False, fontsize=9, labelcolor=INK)
    panel.set_xlabel("Distance from anchored plane (voxels)")
    panel.set_ylabel("Agreement with reference (%)")
    panel.set_title(
        "(d) Internal anchors: agreement vs distance", loc="left", color=INK
    )
    panel.grid(axis="y", color="#e5e7eb", lw=0.8)

    panel = figure.add_subplot(grid[1, 3:6])
    counts = sorted({r["count"] for r in anchor["coverage"]})
    whole = np.asarray(
        [[r["whole"] for r in anchor["coverage"] if r["count"] == c] for c in counts]
    )
    free = [
        [
            r["free_planes"]
            for r in anchor["coverage"]
            if r["count"] == c and r["free_planes"] is not None
        ]
        for c in counts
    ]
    positions = np.arange(len(counts))
    panel.plot(
        positions,
        100 * whole.mean(1),
        color=SERIES[0],
        lw=2,
        marker="o",
        ms=4,
        label="All planes",
    )
    keep = [i for i, values in enumerate(free) if values]
    panel.plot(
        positions[keep],
        [100 * np.mean(free[i]) for i in keep],
        color=SERIES[1],
        lw=2,
        marker="o",
        ms=4,
        label="Unanchored planes only",
    )
    panel.legend(frameon=False, fontsize=9, labelcolor=INK)
    panel.set_xticks(positions, [str(c) for c in counts])
    panel.set_xlabel("Nested axis-0 anchor planes (of 64)")
    panel.set_ylabel("Agreement with reference (%)")
    panel.set_title("(e) Controlled recovery", loc="left", color=INK)
    panel.grid(axis="y", color="#e5e7eb", lw=0.8)
    save(figure, "lr-anchor.png")


def save(figure, name, pad=0.05):
    path = OUTPUT_DIR / name
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=pad,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    print(f"Figure: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
