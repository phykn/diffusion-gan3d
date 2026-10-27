"""Render the LR paper figures from evaluate_lr.py results and cached volumes."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.metrics import read_metrics
from scripts.paper.evaluate_lr import CACHE, DISTANCES, OUTPUT, real_crops
from scripts.paper.make_assets import OUTPUT_DIR, PHASE_RGB, draw_volume
from src.build.data import build_datasets
from src.config.train import load_train_config

# Categorical slots 1-3 of the validated default palette, in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MUTED = "#6b7280"
INK = "#1f2937"
CMAP = ListedColormap(PHASE_RGB)


def main() -> None:
    results = json.loads(OUTPUT.read_text())
    run = PROJECT_ROOT / results["run"]
    best = results["best_step"]
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
    render_volume(volumes[0].numpy())
    render_sections(volumes[0], run)
    render_training(results, run)
    render_anchor(results)


def render_volume(volume: np.ndarray) -> None:
    figure = plt.figure(figsize=(8, 7), facecolor="#f8fafc")
    panel = figure.add_subplot(111, projection="3d")
    draw_volume(panel, volume, PHASE_RGB)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(figure, "02-generated-volume.png", pad=0.03)


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


def render_sections(volume: torch.Tensor, run: Path) -> None:
    real = real_crops(build_datasets(load_train_config(run / "train.yaml"))[0][0])
    labels = real.argmax(1)
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


def render_training(results, run: Path) -> None:
    steps = results["selection"]
    x = [row["step"] for row in steps]
    best = results["best_step"]
    real = results["real"]["half_split_errors"]
    rows = read_metrics(run / "tensorboard")
    figure, panels = plt.subplots(1, 3, figsize=(12, 3.5))

    panel = panels[0]
    panel.plot(x, [row["structure"] for row in steps], color=SERIES[0], lw=2)
    mark_schedule(panel, best)
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
    mark_schedule(panel, best)
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
    mark_schedule(panel, best)
    panel.set_title("(c) Mean critic logit", loc="left", color=INK)
    panel.set_xlabel("Step")
    panel.set_xlim(0, x[-1] * 1.22)
    panel.set_xticks(range(0, x[-1] + 1, 2000))
    figure.tight_layout(w_pad=1.5)
    save(figure, "lr-training.png")


def mark_schedule(panel, best):
    panel.axvspan(0, 5000, color="#e5e7eb", alpha=0.6, lw=0)
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
