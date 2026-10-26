"""Evaluate internal and one-sided anchor continuation."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.diagnostic import generate_with_baseline
from scripts.paper.make_anchor_asset import load_center_roi
from scripts.paper.make_assets import OUTPUT_DIR, PHASE_RGB, ROI_COLOR, SAMPLE_PATH
from scripts.paper.provenance import (
    build_provenance,
    file_record,
    validate_output_paths,
)
from scripts.paper.summary import summarize
from src.build.predict import load_generator, resolve_weights
from src.config.generation import load_generation_settings
from src.evaluate.anchor import (
    measure_boundaries,
    measure_distance_divergence,
    measure_slice_smoothness,
)
from src.evaluate.label import voxel_accuracy
from src.predict.anchor import PlaneAnchor

SEEDS = (0, 1, 2, 3)
INTERNAL_AXES = (0, 1, 2)
DISPLAY_DISTANCES = (0, 1, 4, 16, 64)
OUTPUT = OUTPUT_DIR / "05-boundary-continuation.png"
METADATA = OUTPUT.with_suffix(".json")

INTERNAL = "Generated-control internal"
BOUNDARY = "Generated-control boundary"
EXTERNAL = "External boundary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance", type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = resolve_weights(args.weight)
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    anchor_strength = settings.anchor_strength
    generator = load_generator(weights, device=device)
    margin = generator.default_margin
    provenance = build_provenance(
        weights,
        guidance,
        generation={
            "seeds": list(SEEDS),
            "domain": args.domain,
            "internal_axes": list(INTERNAL_AXES),
            "boundary_axis": 0,
            "anchor_strength": anchor_strength,
            "margin": margin,
        },
        additional_inputs={"training_image": SAMPLE_PATH},
    )
    validate_output_paths(provenance, (OUTPUT, METADATA))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    examples: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    external_anchor = load_center_roi(generator.patch_size)

    def run_case(condition, seed, anchor, axis, index, case_seed):
        torch.manual_seed(case_seed)
        generated, baseline = generate_with_baseline(
            generator,
            (PlaneAnchor(anchor, axis, index),),
            anchor_strength,
            guidance=guidance,
            domain=args.domain,
            margin=margin,
        )
        rows.append(
            measure_case(
                condition,
                seed,
                axis,
                index,
                anchor,
                generated,
                baseline,
                generator.num_phases,
            )
        )
        if seed == 0 and condition != INTERNAL:
            examples[condition] = (anchor.cpu(), generated.cpu())

    for seed in SEEDS:
        torch.manual_seed(40_000 + seed)
        reference = generator.generate(
            anchors=(),
            anchor_strength=0.0,
            guidance=guidance,
            domain=args.domain,
            margin=margin,
        )
        index = generator.patch_size // 2
        for axis in INTERNAL_AXES:
            run_case(
                INTERNAL,
                seed,
                reference.movedim(axis, 0)[index],
                axis,
                index,
                50_000 + seed * len(INTERNAL_AXES) + axis,
            )
        run_case(BOUNDARY, seed, reference[0], 0, 0, 60_000 + seed)
        run_case(EXTERNAL, seed, external_anchor, 0, 0, 70_000 + seed)

    summary = summarize(
        rows,
        (INTERNAL, BOUNDARY, EXTERNAL),
        (
            "pool4_similarity",
            "baseline_pool4_similarity",
            "pool4_gain",
            "pixel_accuracy",
            "baseline_pixel_accuracy",
            "pixel_gain",
            "first_change_ratio",
            "smoothness_p95_ratio",
            "smoothness_peak_ratio",
            "farthest_drift",
            "mean_drift",
        ),
    )
    render(examples, generator.num_phases, OUTPUT)
    METADATA.write_text(
        json.dumps(
            {
                **provenance,
                "metrics": {
                    "pool4_similarity": (
                        "one minus pooled categorical total-variation distance"
                    ),
                    "pixel_accuracy": "exact agreement on the supplied plane",
                    "baseline_pool4_similarity": (
                        "same-noise baseline pool4 similarity to the supplied plane"
                    ),
                    "pool4_gain": (
                        "conditioned minus same-noise baseline pool4 similarity"
                    ),
                    "baseline_pixel_accuracy": (
                        "same-noise baseline exact agreement with the supplied plane"
                    ),
                    "pixel_gain": (
                        "conditioned minus same-noise baseline exact agreement"
                    ),
                    "first_change_ratio": (
                        "anchor-adjacent change divided by ordinary slice change"
                    ),
                    "smoothness_p95_ratio": (
                        "p95 second-difference divided by same-RNG baseline"
                    ),
                    "smoothness_peak_ratio": (
                        "maximum second-difference divided by same-RNG baseline"
                    ),
                    "farthest_drift": (
                        "same-RNG disagreement at the farthest available distance"
                    ),
                    "mean_drift": "mean same-RNG disagreement over the complete axis",
                },
                "raw": rows,
                "summary": summary,
                "output": file_record(OUTPUT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary)
    print(f"Figure  : {OUTPUT.resolve()}")
    print(f"Metadata: {METADATA.resolve()}")


def measure_case(
    condition: str,
    seed: int,
    axis: int,
    index: int,
    anchor: torch.Tensor,
    generated: torch.Tensor,
    baseline: torch.Tensor,
    num_phases: int,
) -> dict[str, object]:
    generated_plane = generated.movedim(axis, 0)[index]
    baseline_plane = baseline.movedim(axis, 0)[index]
    boundary = measure_boundaries(generated, (index,), axis)
    smoothness = measure_slice_smoothness(generated, (index,), axis, baseline)
    profile = measure_distance_divergence(generated, baseline, (index,), axis)
    valid = [value for value in profile if value is not None]
    pool4 = pooled_similarity(generated_plane, anchor, num_phases, pool_size=4)
    baseline_pool4 = pooled_similarity(baseline_plane, anchor, num_phases, pool_size=4)
    pixel = voxel_accuracy(generated_plane, anchor)
    baseline_pixel = voxel_accuracy(baseline_plane, anchor)
    return {
        "condition": condition,
        "seed": seed,
        "axis": axis,
        "index": index,
        "pool4_similarity": pool4,
        "baseline_pool4_similarity": baseline_pool4,
        "pool4_gain": pool4 - baseline_pool4,
        "pixel_accuracy": pixel,
        "baseline_pixel_accuracy": baseline_pixel,
        "pixel_gain": pixel - baseline_pixel,
        "first_change_ratio": boundary.change_ratio,
        "smoothness_p95_ratio": smoothness.p95_ratio,
        "smoothness_peak_ratio": smoothness.max_ratio,
        "farthest_drift": valid[-1] if valid else None,
        "mean_drift": float(np.mean(valid)) if valid else None,
    }


def pooled_similarity(
    generated: torch.Tensor,
    target: torch.Tensor,
    num_phases: int,
    *,
    pool_size: int,
) -> float:
    def pooled(labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels.to(torch.long), num_phases).permute(2, 0, 1)
        return F.avg_pool2d(one_hot.to(torch.float32), pool_size, stride=pool_size)

    distance = 0.5 * (pooled(generated) - pooled(target)).abs().sum(dim=0)
    return 1.0 - float(distance.mean())


def render(
    examples: dict[str, tuple[torch.Tensor, torch.Tensor]],
    num_phases: int,
    output: Path,
) -> None:
    colors = PHASE_RGB[:num_phases]
    cmap = ListedColormap(colors)
    depth = len(examples[BOUNDARY][1])
    # Smaller models have fewer slices; skip distances beyond the volume.
    distances = tuple(
        dict.fromkeys(d for d in (*DISPLAY_DISTANCES, depth - 1) if d < depth)
    )
    columns = 1 + len(distances)
    figure, panels = plt.subplots(
        2, columns, figsize=(15.5 * columns / 7, 4.8), squeeze=False
    )
    for row, condition in enumerate((BOUNDARY, EXTERNAL)):
        target, slices = examples[condition]
        images = (target, *(slices[distance] for distance in distances))
        titles = (
            ("Generated-control input" if condition == BOUNDARY else "External input"),
            *(
                f"Generated d={d}" if index == 0 else f"d={d}"
                for index, d in enumerate(distances)
            ),
        )
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            panel = panels[row, column]
            panel.imshow(
                image.numpy(),
                cmap=cmap,
                vmin=-0.5,
                vmax=num_phases - 0.5,
                interpolation="nearest",
            )
            panel.set_title(title, fontsize=10)
            panel.axis("off")
        panels[row, 0].add_patch(
            Rectangle(
                (-0.5, -0.5),
                target.shape[1],
                target.shape[0],
                fill=False,
                edgecolor=ROI_COLOR,
                linewidth=2.2,
            )
        )
    figure.suptitle("One-sided 3D continuation from a supplied boundary section")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def print_summary(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            f"{row['condition']} (n={row['samples']}): "
            f"pool4={row['baseline_pool4_similarity_mean']:.2%}->"
            f"{row['pool4_similarity_mean']:.2%}, "
            f"pixel={row['pixel_accuracy_mean']:.2%}, "
            f"first={row['first_change_ratio_mean']:.2f}x, "
            f"p95={row['smoothness_p95_ratio_mean']:.2f}x, "
            f"far={row['farthest_drift_mean']:.2%}"
        )


if __name__ == "__main__":
    main()
