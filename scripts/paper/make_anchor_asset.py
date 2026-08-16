import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
)

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings

AXIS = 0
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
    )
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance", type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weight.resolve()
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    generator = load_generator(weights, device=device)
    margin = generator.default_margin
    provenance = build_provenance(
        weights,
        guidance,
        generation={
            "seed": SEED,
            "domain": args.domain,
            "axis": AXIS,
            "margin": margin,
            "source_roi_left_top": list(ROI_POSITIONS[1]),
            "source_crop_size": CROP_SIZE,
        },
        reference=SAMPLE_PATH,
    )
    output = OUTPUT_DIR / "04-anchor-conditioning.png"
    metadata = output.with_suffix(".json")
    validate_output_paths(provenance, (output, metadata))
    anchor = load_center_roi(generator.patch_size)
    index = generator.patch_size // 2
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("\nPaper anchor example")
    print("---------------------")
    print(f"Weights : {weights.resolve()}")
    print(f"Device  : {device}")
    print(f"Guidance: {guidance}")
    print(f"Anchor  : axis {AXIS}, index {index}, shape {tuple(anchor.shape)}")
    print("Status  : generating...", flush=True)
    volume = generator.generate(
        anchors=(PlaneAnchor(image=anchor, axis=AXIS, index=index),),
        guidance=guidance,
        domain=args.domain,
        margin=margin,
    )
    generated = volume.select(AXIS, index)
    matches = int((generated == anchor).sum())
    total = anchor.numel()
    accuracy = matches / total
    print("Status  : complete")
    print(f"Match   : {matches} / {total} ({accuracy:.4%})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_result(anchor, generated, volume, output)
    metadata.write_text(
        json.dumps(
            {
                **provenance,
                "seed": SEED,
                "anchor": {
                    "axis": AXIS,
                    "index": index,
                    "shape": list(anchor.shape),
                    "source": str(SAMPLE_PATH.resolve()),
                    "source_roi_left_top": list(ROI_POSITIONS[1]),
                    "source_crop_size": CROP_SIZE,
                },
                "output": {
                    **file_record(output),
                    "shape": list(volume.shape),
                    "dtype": str(volume.dtype),
                },
                "anchor_match": {
                    "matched_voxels": matches,
                    "total_voxels": total,
                    "accuracy": accuracy,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Figure  : {output.resolve()}")
    print(f"Metadata: {metadata.resolve()}")


def load_center_roi(size: int) -> torch.Tensor:
    left, top = ROI_POSITIONS[1]
    with Image.open(SAMPLE_PATH) as image:
        labels = np.asarray(image).copy()
    crop = labels[top : top + CROP_SIZE, left : left + CROP_SIZE]
    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError("center paper ROI does not have the configured crop size.")
    tensor = torch.from_numpy(crop).to(torch.float32)[None, None]
    resized = F.interpolate(tensor, size=(size, size), mode="nearest")
    return resized[0, 0].to(torch.long)


def render_result(
    anchor: torch.Tensor,
    generated: torch.Tensor,
    volume: torch.Tensor,
    output: Path,
) -> None:
    colors = [to_rgb(color) for color in PHASE_COLORS]
    cmap = ListedColormap(colors)
    figure = plt.figure(figsize=(12.4, 4.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 1.8),
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.90,
        wspace=0.12,
    )
    panels = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    )
    for panel, image, title in zip(
        panels,
        (anchor, generated),
        ("(a) Supplied center section", "(b) Generated center section"),
        strict=True,
    ):
        panel.imshow(
            image.numpy(),
            cmap=cmap,
            vmin=-0.5,
            vmax=len(colors) - 0.5,
            interpolation="nearest",
        )
        panel.set_title(title, fontsize=13, pad=8)
        panel.axis("off")
    panels[0].add_patch(
        Rectangle(
            (-0.5, -0.5),
            anchor.shape[1],
            anchor.shape[0],
            fill=False,
            edgecolor=ROI_COLOR,
            linewidth=2.0,
        )
    )

    volume_panel = figure.add_subplot(grid[0, 2], projection="3d")
    draw_volume(volume_panel, volume.numpy(), colors)
    size = volume.shape[0]
    volume_panel.set_title(f"(c) Anchored {size}³ volume", fontsize=13, pad=2)
    figure.savefig(
        output,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
