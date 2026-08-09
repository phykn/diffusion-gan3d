"""Generate the anchored 3x3x3 scale-up example used in PAPER.md."""

import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from make_anchor_asset import AXIS, SEED, load_center_roi
from make_assets import OUTPUT_DIR, PHASE_COLORS, ROI_COLOR, draw_volume

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.generate import DEFAULT_SCALE_OVERLAP, ScaledGenerator
from src.train.weights import find_weights

BLOCKS = (3, 3, 3)
OVERLAP = DEFAULT_SCALE_OVERLAP


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = find_weights(PROJECT_ROOT / "run")
    generator = load_generator(weights, device=device)
    if not generator.anchor_enabled:
        raise ValueError("selected weights were trained with anchors disabled.")

    anchor = load_center_roi(generator.patch_size)
    anchor_index = generator.patch_size // 2
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("\nPaper scale-up example")
    print("-----------------------")
    print(f"Weights : {weights.resolve()}")
    print(f"Device  : {device}")
    print(f"Blocks  : {BLOCKS}")
    print(f"Overlap : {OVERLAP}")
    print("Status  : generating anchored base...", flush=True)
    base = generator.generate(
        anchors=(PlaneAnchor(image=anchor, axis=AXIS, index=anchor_index),)
    )

    scaled = ScaledGenerator(generator)
    shape = tuple(generator.patch_size * count for count in BLOCKS)
    plan = scaled.plan(shape, OVERLAP)
    print(f"Shape   : {shape}")
    print(f"Tiles   : {plan.tile_count}")
    print("Status  : scaling...", flush=True)
    start_time = perf_counter()
    volume = scaled.generate(
        blocks=BLOCKS,
        overlap=OVERLAP,
        base=base,
        progress=False,
    )
    elapsed = perf_counter() - start_time
    print(f"Status  : complete ({elapsed:.1f} s)")

    start = tuple((size - generator.patch_size) // 2 for size in volume.shape)
    global_index = start[AXIS] + anchor_index
    center_plane = volume.select(AXIS, global_index)
    embedded = center_plane[
        start[1] : start[1] + generator.patch_size,
        start[2] : start[2] + generator.patch_size,
    ]
    matches = int((embedded == anchor).sum())
    total = anchor.numel()
    shell = plan.base_shell
    core = slice(shell, -shell) if shell else slice(None)
    core_matches = int((embedded[core, core] == anchor[core, core]).sum())
    core_total = anchor[core, core].numel()
    embedded_base = volume[
        start[0] : start[0] + generator.patch_size,
        start[1] : start[1] + generator.patch_size,
        start[2] : start[2] + generator.patch_size,
    ]
    base_core_matches = int(
        (embedded_base[core, core, core] == base[core, core, core]).sum()
    )
    base_core_total = base[core, core, core].numel()
    print(f"Anchor  : {matches} / {total} ({matches / total:.4%})")
    print(
        f"Plane core: {core_matches} / {core_total} "
        f"({core_matches / core_total:.4%})"
    )
    print(
        f"Base core : {base_core_matches} / {base_core_total} "
        f"({base_core_matches / base_core_total:.4%})"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "05-scale-up.png"
    render_result(
        anchor=anchor,
        center_plane=center_plane,
        volume=volume,
        start=start,
        shell=shell,
        output=output,
    )
    print(f"Figure  : {output.resolve()}")


def render_result(
    anchor: torch.Tensor,
    center_plane: torch.Tensor,
    volume: torch.Tensor,
    start: tuple[int, int, int],
    shell: int,
    output: Path,
) -> None:
    colors = [to_rgb(color) for color in PHASE_COLORS]
    cmap = ListedColormap(colors)
    figure = plt.figure(figsize=(13.3, 4.7), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1, 1.35, 1.8),
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.90,
        wspace=0.12,
    )

    anchor_panel = figure.add_subplot(grid[0, 0])
    anchor_panel.imshow(
        anchor.numpy(),
        cmap=cmap,
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        interpolation="nearest",
    )
    anchor_panel.set_title(
        f"(a) {anchor.shape[0]}×{anchor.shape[1]} center-plane anchor",
        fontsize=13,
        pad=8,
    )
    anchor_panel.axis("off")
    anchor_panel.add_patch(
        Rectangle(
            (-0.5, -0.5),
            anchor.shape[1],
            anchor.shape[0],
            fill=False,
            edgecolor=ROI_COLOR,
            linewidth=2.0,
        )
    )

    plane_panel = figure.add_subplot(grid[0, 1])
    plane_panel.imshow(
        center_plane.numpy(),
        cmap=cmap,
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        interpolation="nearest",
    )
    plane_panel.add_patch(
        Rectangle(
            (start[2] - 0.5, start[1] - 0.5),
            anchor.shape[1],
            anchor.shape[0],
            fill=False,
            edgecolor=ROI_COLOR,
            linewidth=2.0,
        )
    )
    if shell:
        plane_panel.add_patch(
            Rectangle(
                (start[2] + shell - 0.5, start[1] + shell - 0.5),
                anchor.shape[1] - 2 * shell,
                anchor.shape[0] - 2 * shell,
                fill=False,
                edgecolor="#2563EB",
                linewidth=1.8,
                linestyle="--",
            )
        )
    plane_panel.set_title(
        f"(b) {center_plane.shape[0]}×{center_plane.shape[1]} center section",
        fontsize=13,
        pad=8,
    )
    plane_panel.axis("off")

    volume_panel = figure.add_subplot(grid[0, 2], projection="3d")
    draw_volume(volume_panel, volume.numpy(), colors)
    volume_panel.set_title(f"(c) {volume.shape[0]}³ output", fontsize=13, pad=2)
    figure.savefig(
        output,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
