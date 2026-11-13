import argparse
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

from paper.scripts.common import cache_dir, first_image, read_metrics, real_crops
from paper.scripts.evaluate_anode import lr_grid, real_volume_paths
from paper.scripts.evaluate_lr import DISTANCES
from paper.scripts.evaluate_sr import upsample_linear, upsample_nearest
from src.config.io import load_config, project_path
from src.stats.sections import label_sections
from src.storage import load_volume

PHASE_RGB = [to_rgb(color) for color in ("#000000", "#9E9E9E")]
# Pore, graphite, silicon; the third phase light, as in the simulated slices.
ANODE_RGB = [*PHASE_RGB, to_rgb("#F2F2F2")]
ROI_COLOR = "#F97316"
# (left, top) of the example crops drawn on the training image, beside the
# external anchor crop.
ROI_POSITIONS = ((18, 16), (536, 24))
# Categorical slots 1-3 of the validated default palette, in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MUTED = "#6b7280"
INK = "#1f2937"
CMAP = ListedColormap(PHASE_RGB)
ANODE_CMAP = ListedColormap(ANODE_RGB)
# Moving-average window of the critic logits, in logged steps.
LOGIT_WINDOW = 200


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the paper figures from the evaluate_*.py results "
        "and their caches."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Folder with lr-, sr- and anode-results.json and their cache/.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Folder for the PNG figures."
    )
    args = parser.parse_args(argv)
    render_figures(args.results, args.out)


def render_figures(root, out):
    out.mkdir(parents=True, exist_ok=True)

    results = read_results(root, "lr")
    run = project_path(results["run"])
    best = results["best_step"]
    config = load_config(run / "train.yaml")
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
    cache = cache_dir(root, run)
    volumes = torch.load(cache / f"step{best}.pt", weights_only=True)
    external = torch.load(cache / "anchor-external.pt", weights_only=True)
    render_training_image(config, external["crop"], out)
    render_volume(volumes[0].numpy(), out / "02-generated-volume.png")
    render_sections(volumes[0], config, out)
    render_training(results, run, config.burn_in.steps, out)
    render_anchor(results, external, out)
    render_sr(read_results(root, "sr"), root, out)
    render_anode(read_results(root, "anode"), root, out)


def read_results(root, stage):
    return json.loads((root / f"{stage}-results.json").read_text(encoding="utf-8"))


def render_training_image(config, external_crop, out):
    labels = first_image(config)[1]
    rgb = np.rint(255 * np.asarray(PHASE_RGB)[labels]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    size = config.data.crop_size
    for left, top in (ROI_POSITIONS[0], external_crop, ROI_POSITIONS[1]):
        draw.rectangle(
            (left, top, left + size - 1, top + size - 1), outline=ROI_COLOR, width=4
        )
    path = out / "01-training-data.png"
    image.save(path)
    print(f"Figure: {path}")


def render_volume(volume, path):
    figure = plt.figure(figsize=(8, 7), facecolor="#f8fafc")
    panel = figure.add_subplot(111, projection="3d")
    draw_volume(panel, volume, PHASE_RGB)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(figure, path, pad=0.03)


def draw_volume(
    panel,
    volume,
    colors,
):
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
    panel,
    volume,
    rgba,
    axis,
    index,
    first,
    second,
    light,
):
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


def draw_box(panel, size):
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


def show(panel, image, title=None, cmap=CMAP):
    panel.imshow(
        np.asarray(image),
        cmap=cmap,
        vmin=-0.5,
        vmax=cmap.N - 0.5,
        interpolation="nearest",
    )
    if title:
        panel.set_title(title, fontsize=10, color=INK, pad=5)
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_visible(False)


def render_sections(volume, config, out):
    labels = label_sections(real_crops(config, config.data.lo_res_size)).argmax(1)
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
    save(figure, out / "03-generated-slices.png")


def render_training(results, run, burn_in, out):
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

    plot_training_logits(panels[2], rows, best, burn_in, x[-1], 2000)
    figure.tight_layout(w_pad=1.5)
    save(figure, out / "lr-training.png")


def plot_training_logits(panel, rows, best, burn_in, last_step, tick):
    plot_logits(panel, rows)
    mark_schedule(panel, best, burn_in)
    panel.set_title("(c) Mean critic logit", loc="left", color=INK)
    panel.set_xlabel("Step")
    panel.set_xlim(0, last_step * 1.22)
    panel.set_xticks(range(0, last_step + 1, tick))


def plot_logits(panel, rows):
    """Mean critic logit of real and generated pairs over the three planes."""
    names = {tag for row in rows for tag in row if tag.startswith("score/")}
    for color, (kind, label) in zip(SERIES, (("real", "Real"), ("fake", "Generated"))):
        tags = sorted(tag for tag in names if tag.endswith(f"/{kind}"))
        points = [
            (row["step"], np.mean([row[t] for t in tags]))
            for row in rows
            if all(row.get(t) is not None for t in tags)
        ]
        s, v = np.asarray(points).T
        window = min(LOGIT_WINDOW, len(v))
        smooth = np.convolve(v, np.ones(window) / window, mode="valid")
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


def mark_schedule(panel, best, burn_in):
    panel.axvspan(0, burn_in, color="#e5e7eb", alpha=0.6, lw=0)
    panel.axvline(best, color=MUTED, lw=1, ls="--")
    panel.grid(axis="y", color="#e5e7eb", lw=0.8)


def render_anchor(results, external, out):
    anchor = results["anchor"]
    plane = external["plane"]
    size = plane.shape[0]
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
    # Small volumes leave out the far sections.
    far = [d for d in (2, 8, 24) if index + d < size]
    for column, d in zip((3, 4, 5), far):
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
    panel.set_xlabel(f"Nested axis-0 anchor planes (of {size})")
    panel.set_ylabel("Agreement with reference (%)")
    panel.set_title("(e) Controlled recovery", loc="left", color=INK)
    panel.grid(axis="y", color="#e5e7eb", lw=0.8)
    save(figure, out / "lr-anchor.png")


def render_sr(results, root, out):
    run = project_path(results["run"])
    config = load_config(run / "train.yaml")
    best = results["best_step"]
    cache = cache_dir(root, run)
    low = torch.load(cache / "lr.pt", weights_only=True)
    high = torch.load(cache / f"step{best}.pt", weights_only=True)
    render_sr_training(results, run, config.burn_in.steps, out)
    render_sr_sections(low[0], high[0], config, out)
    render_volume(high[0].numpy(), out / "sr-volume.png")


def render_sr_training(results, run, burn_in, out):
    steps = results["selection"]
    x = [row["step"] for row in steps]
    best = results["best_step"]
    figure, panels = plt.subplots(1, 3, figsize=(12, 3.5))
    for panel, key, title, label in (
        (
            panels[0],
            "structure",
            "(a) Structure score (selection)",
            "Mean error / real split error",
        ),
        (panels[1], "lr_mse", "(b) Mismatch with LR input", "Downsampled HR vs LR MSE"),
    ):
        panel.plot(x, [row[key] for row in steps], color=SERIES[0], lw=2)
        mark_schedule(panel, best, burn_in)
        panel.set_title(title, loc="left", color=INK)
        panel.set_xlabel("Step")
        panel.set_ylabel(label)
    plot_training_logits(
        panels[2], read_metrics(run / "tensorboard"), best, burn_in, x[-1], 1000
    )
    figure.tight_layout(w_pad=1.5)
    save(figure, out / "sr-training.png")


def render_sr_sections(low, high, config, out):
    labels = label_sections(real_crops(config, config.data.hi_res_size)).argmax(1)
    scale = config.scale_factor
    volumes = (
        ("LR, nearest", upsample_nearest(low, scale)),
        ("LR, trilinear", upsample_linear(low, scale)),
        ("SR", high),
    )
    center = high.shape[0] // 2
    figure, panels = plt.subplots(4, 3, figsize=(8.4, 11.4))
    for column in range(3):
        show(panels[0, column], labels[column], f"Real crop {column + 1}")
    for row, (name, volume) in enumerate(volumes, 1):
        for column, (plane, section) in enumerate(
            zip(
                ("xy", "xz", "yz"),
                (volume[center], volume[:, center], volume[:, :, center]),
            )
        ):
            show(panels[row, column], section, f"{name} {plane}")
    figure.tight_layout(pad=0.6, h_pad=1.0)
    save(figure, out / "sr-sections.png")


def render_anode(results, root, out):
    run = project_path(results["run"])
    config = load_config(project_path(results["lr_run"]) / "train.yaml")
    cache = cache_dir(root, run)
    probs = torch.load(cache / "lr.pt", weights_only=True)
    high = torch.load(cache / "hr.pt", weights_only=True)
    real = [load_volume(p).to(torch.uint8) for p in real_volume_paths(config.data)]
    scale = results["source_pixels_per_lr_voxel"]
    low = probs[0].argmax(0).to(torch.uint8)
    phases = config.data.num_phases
    render_anode_profile(results, scale, out)
    real_low = lr_grid(real[0], scale, phases)
    render_anode_sections(real[0], real_low, low, high[0], scale, out)
    render_anode_volume(real[0].numpy(), high[0].numpy(), out)


def render_anode_profile(results, scale, out):
    names = ("Pore", "Graphite", "Silicon")
    figure, panels = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    lr, sr = results["lr"]["profile"], results["sr"]["profile"]
    series = (
        ("Real", sr["real"], None, 1, INK),
        ("LR", lr["generated"], lr["generated_std"], scale, SERIES[0]),
        ("SR", sr["generated"], sr["generated_std"], 1, SERIES[1]),
    )
    for phase, panel in enumerate(panels):
        for label, mean, std, spacing, color in series:
            mean = np.asarray(mean[phase])
            depth = (np.arange(len(mean)) + 0.5) * spacing
            panel.plot(depth, mean, color=color, lw=1.8, label=label)
            if std is not None:
                std = np.asarray(std[phase])
                panel.fill_between(
                    depth, mean - std, mean + std, color=color, alpha=0.15, lw=0
                )
        panel.set_title(f"({'abc'[phase]}) {names[phase]}", loc="left", color=INK)
        panel.set_xlabel("Depth z (source pixels)")
        panel.grid(axis="y", color="#e5e7eb", lw=0.8)
    panels[0].set_ylabel("Fraction of the xy section")
    panels[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(w_pad=1.5)
    save(figure, out / "anode-profile.png")


def render_anode_sections(real, real_low, low, high, scale, out):
    """Real, LR and SR sections: xy near both faces, and the two side planes."""
    rows = (
        ("Real", real, 1),
        ("Real, LR grid", real_low, scale),
        ("LR", low, scale),
        ("SR", high, 1),
    )
    figure, panels = plt.subplots(4, 4, figsize=(11.2, 11.6))
    for row, (name, volume, step) in enumerate(rows):
        depth = volume.shape[0]
        center = volume.shape[1] // 2
        sections = (
            (f"xy, z = {depth // 4 * step}", volume[depth // 4]),
            (f"xy, z = {3 * depth // 4 * step}", volume[3 * depth // 4]),
            ("xz", volume[:, center]),
            ("yz", volume[:, :, center]),
        )
        for column, (title, section) in enumerate(sections):
            panel = panels[row, column]
            show(panel, section, f"{name}, {title}", cmap=ANODE_CMAP)
    figure.tight_layout(pad=0.6, h_pad=1.0)
    save(figure, out / "anode-sections.png")


def render_anode_volume(real, high, out):
    """Real and SR volumes with depth z drawn upward and one octant removed."""
    figure = plt.figure(figsize=(12, 6), facecolor="#f8fafc")
    for column, (title, volume) in enumerate((("Real", real), ("Generated", high))):
        panel = figure.add_subplot(1, 2, column + 1, projection="3d")
        panel.set_facecolor(figure.get_facecolor())
        draw_volume(panel, np.ascontiguousarray(volume.transpose(2, 1, 0)), ANODE_RGB)
        figure.text(0.25 + 0.5 * column, 0.97, title, ha="center", color=INK)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0)
    save(figure, out / "anode-volume.png", pad=0.03)


def save(figure, path, pad=0.05):
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=pad,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    print(f"Figure: {path}")


if __name__ == "__main__":
    main()
