"""Generate the anchored 3x3x3 scale-up example used in PAPER.md."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from make_anchor_asset import AXIS, SEED, load_center_roi
from make_assets import (
    CROP_SIZE,
    OUTPUT_DIR,
    PHASE_COLORS,
    ROI_COLOR,
    ROI_POSITIONS,
    SAMPLE_PATH,
    draw_volume,
)
from provenance import (
    build_provenance,
    file_record,
    validate_output_paths,
    verify_provenance_inputs,
)

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.scale import DEFAULT_SCALE_OVERLAP, ScaledGenerator

BLOCKS = (3, 3, 3)
OVERLAP = DEFAULT_SCALE_OVERLAP


@dataclass(frozen=True)
class ScaleAssessment:
    start: tuple[int, int, int]
    shell: int
    center_plane: torch.Tensor
    anchor_matches: int
    anchor_total: int
    plane_core_matches: int
    plane_core_total: int
    base_core_matches: int
    base_core_total: int

    def metadata(self) -> dict[str, int | float]:
        return {
            "anchor_matched_voxels": self.anchor_matches,
            "anchor_total_voxels": self.anchor_total,
            "anchor_accuracy": self.anchor_matches / self.anchor_total,
            "plane_core_matched_voxels": self.plane_core_matches,
            "plane_core_total_voxels": self.plane_core_total,
            "base_core_matched_voxels": self.base_core_matches,
            "base_core_total_voxels": self.base_core_total,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weight.resolve()
    provenance = build_provenance(
        weights,
        args.guidance_scale,
        generation={
            "seed": SEED,
            "axis": AXIS,
            "blocks": list(BLOCKS),
            "overlap": OVERLAP,
            "source_roi_left_top": list(ROI_POSITIONS[1]),
            "source_crop_size": CROP_SIZE,
        },
        reference=SAMPLE_PATH,
        source_files=(__file__,),
    )
    output = OUTPUT_DIR / "05-scale-up.png"
    metadata = output.with_suffix(".json")
    validate_output_paths(provenance, (output, metadata))
    generator = load_generator(weights, device=device)
    if not generator.anchor_enabled:
        raise ValueError("selected weights were trained with anchors disabled.")

    anchor = load_center_roi(generator.patch_size)
    verify_provenance_inputs(provenance)
    anchor_index = generator.patch_size // 2
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("\nPaper scale-up example")
    print("-----------------------")
    print(f"Weights : {weights.resolve()}")
    print(f"Device  : {device}")
    print(f"Guidance: {args.guidance_scale}")
    print(f"Blocks  : {BLOCKS}")
    print(f"Overlap : {OVERLAP}")
    print("Status  : generating anchored base...", flush=True)
    base = generator.generate(
        anchors=(PlaneAnchor(image=anchor, axis=AXIS, index=anchor_index),),
        guidance_scale=args.guidance_scale,
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
        guidance_scale=args.guidance_scale,
    )
    elapsed = perf_counter() - start_time
    print(f"Status  : complete ({elapsed:.1f} s)")

    assessment = assess_scale(
        volume,
        base,
        anchor,
        anchor_index,
        plan.base_shell,
    )
    print_assessment(assessment)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_result(
        anchor=anchor,
        volume=volume,
        assessment=assessment,
        output=output,
    )
    metadata.write_text(
        json.dumps(
            {
                **provenance,
                "seed": SEED,
                "scale_plan": {
                    "blocks": list(BLOCKS),
                    "overlap": OVERLAP,
                    "tile_count": plan.tile_count,
                    "base_shell": assessment.shell,
                },
                "anchor": {
                    "axis": AXIS,
                    "index": anchor_index,
                    "shape": list(anchor.shape),
                },
                "base": {
                    "shape": list(base.shape),
                    "dtype": str(base.dtype),
                },
                "output": {
                    **file_record(output),
                    "shape": list(volume.shape),
                    "dtype": str(volume.dtype),
                    "elapsed_seconds": elapsed,
                },
                "quality": assessment.metadata(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Figure  : {output.resolve()}")
    print(f"Metadata: {metadata.resolve()}")


def assess_scale(
    volume: torch.Tensor,
    base: torch.Tensor,
    anchor: torch.Tensor,
    anchor_index: int,
    shell: int,
) -> ScaleAssessment:
    patch_size = base.shape[0]
    if base.shape != (patch_size,) * 3 or anchor.shape != (patch_size,) * 2:
        raise ValueError("base and anchor must use the same cubic patch size.")
    start = tuple((size - patch_size) // 2 for size in volume.shape)
    global_index = start[AXIS] + anchor_index
    center_plane = volume.select(AXIS, global_index)
    embedded = center_plane[
        start[1] : start[1] + patch_size,
        start[2] : start[2] + patch_size,
    ]
    core = slice(shell, -shell) if shell else slice(None)
    embedded_base = volume[
        start[0] : start[0] + patch_size,
        start[1] : start[1] + patch_size,
        start[2] : start[2] + patch_size,
    ]
    return ScaleAssessment(
        start=start,
        shell=shell,
        center_plane=center_plane,
        anchor_matches=int((embedded == anchor).sum()),
        anchor_total=anchor.numel(),
        plane_core_matches=int((embedded[core, core] == anchor[core, core]).sum()),
        plane_core_total=anchor[core, core].numel(),
        base_core_matches=int(
            (embedded_base[core, core, core] == base[core, core, core]).sum()
        ),
        base_core_total=base[core, core, core].numel(),
    )


def print_assessment(assessment: ScaleAssessment) -> None:
    print(
        f"Anchor  : {assessment.anchor_matches} / {assessment.anchor_total} "
        f"({assessment.anchor_matches / assessment.anchor_total:.4%})"
    )
    print(
        f"Plane core: {assessment.plane_core_matches} / "
        f"{assessment.plane_core_total} "
        f"({assessment.plane_core_matches / assessment.plane_core_total:.4%})"
    )
    print(
        f"Base core : {assessment.base_core_matches} / "
        f"{assessment.base_core_total} "
        f"({assessment.base_core_matches / assessment.base_core_total:.4%})"
    )


def render_result(
    anchor: torch.Tensor,
    volume: torch.Tensor,
    assessment: ScaleAssessment,
    output: Path,
) -> None:
    center_plane = assessment.center_plane
    start = assessment.start
    shell = assessment.shell
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
