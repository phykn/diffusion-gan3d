"""Evaluate internal and one-sided anchor continuation."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from make_anchor_asset import load_center_roi
from make_assets import OUTPUT_DIR, PHASE_COLORS, ROI_COLOR, SAMPLE_PATH
from provenance import build_provenance, file_record, validate_output_paths

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
    weights = args.weight.resolve()
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
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
            "anchor_strength": DEFAULT_ANCHOR_STRENGTH,
            "margin": margin,
        },
        additional_inputs={"training_image": SAMPLE_PATH},
    )
    validate_output_paths(provenance, (OUTPUT, METADATA))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    examples: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    external_anchor = load_center_roi(generator.patch_size)
    for seed in SEEDS:
        set_seed(40_000 + seed, device)
        reference = generator.generate(
            anchors=(),
            anchor_strength=0.0,
            guidance=guidance,
            domain=args.domain,
            margin=margin,
        )
        for axis in INTERNAL_AXES:
            index = generator.patch_size // 2
            anchor = reference.movedim(axis, 0)[index]
            generated, baseline = generate_pair(
                generator,
                anchor,
                axis,
                index,
                seed=50_000 + seed * len(INTERNAL_AXES) + axis,
                domain=args.domain,
                guidance=guidance,
                margin=margin,
            )
            rows.append(
                measure_case(
                    INTERNAL,
                    seed,
                    axis,
                    index,
                    anchor,
                    generated,
                    baseline,
                    generator.num_phases,
                )
            )

        generated, baseline = generate_pair(
            generator,
            reference[0],
            axis=0,
            index=0,
            seed=60_000 + seed,
            domain=args.domain,
            guidance=guidance,
            margin=margin,
        )
        rows.append(
            measure_case(
                BOUNDARY,
                seed,
                0,
                0,
                reference[0],
                generated,
                baseline,
                generator.num_phases,
            )
        )
        if seed == 0:
            examples[BOUNDARY] = (reference[0].cpu(), generated.cpu())

        generated, baseline = generate_pair(
            generator,
            external_anchor,
            axis=0,
            index=0,
            seed=70_000 + seed,
            domain=args.domain,
            guidance=guidance,
            margin=margin,
        )
        rows.append(
            measure_case(
                EXTERNAL,
                seed,
                0,
                0,
                external_anchor,
                generated,
                baseline,
                generator.num_phases,
            )
        )
        if seed == 0:
            examples[EXTERNAL] = (external_anchor.cpu(), generated.cpu())

    summary = summarize(rows)
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


def generate_pair(
    generator,
    anchor: torch.Tensor,
    axis: int,
    index: int,
    *,
    seed: int,
    domain: int,
    guidance: float,
    margin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed, generator.device)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if generator.device.type == "cuda" else None
    )
    condition = (PlaneAnchor(anchor, axis, index),)
    generated = generator.generate(
        anchors=condition,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        guidance=guidance,
        domain=domain,
        margin=margin,
    )
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    baseline = generator.generate(
        anchors=(),
        anchor_strength=0.0,
        guidance=guidance,
        domain=domain,
        margin=margin,
    )
    return generated, baseline


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
    boundary = measure_boundaries(generated, (index,), axis, num_phases)
    smoothness = measure_slice_smoothness(generated, (index,), axis, baseline)
    profile = measure_distance_divergence(
        generated,
        baseline,
        (index,),
        axis,
        generated.shape[axis] - 1,
    )
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


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
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
    )
    result = []
    for condition in (INTERNAL, BOUNDARY, EXTERNAL):
        selected = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {
            "condition": condition,
            "samples": len(selected),
        }
        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in selected if row[metric] is not None],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(values.mean()) if values.size else None
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else None
            )
        result.append(summary)
    return result


def render(
    examples: dict[str, tuple[torch.Tensor, torch.Tensor]],
    num_phases: int,
    output: Path,
) -> None:
    colors = [to_rgb(color) for color in PHASE_COLORS[:num_phases]]
    cmap = ListedColormap(colors)
    figure, panels = plt.subplots(2, 7, figsize=(15.5, 4.8), squeeze=False)
    for row, condition in enumerate((BOUNDARY, EXTERNAL)):
        target, volume = examples[condition]
        slices = volume.movedim(0, 0)
        distances = (*DISPLAY_DISTANCES, len(slices) - 1)
        images = (target, *(slices[distance] for distance in distances))
        titles = (
            ("Generated-control input" if condition == BOUNDARY else "External input"),
            "Generated d=0",
            "d=1",
            "d=4",
            "d=16",
            "d=64",
            f"d={len(slices) - 1}",
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


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


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
