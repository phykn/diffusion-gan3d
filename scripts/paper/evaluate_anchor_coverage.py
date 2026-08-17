"""Measure controlled 3D recovery as nested reference planes are added."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from make_assets import OUTPUT_DIR
from provenance import build_provenance, file_record, validate_output_paths

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import voxel_accuracy

COUNTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
AXIS = 0
REFERENCE_SEED = 10_000
GENERATION_SEED = 0
OUTPUT = OUTPUT_DIR / "04-anchor-coverage.png"
METADATA = OUTPUT.with_suffix(".json")


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
    anchor_strength = settings.anchor_strength
    generator = load_generator(weights, device=device)
    size = generator.patch_size
    if COUNTS[-1] != size:
        raise ValueError(
            f"coverage counts require a {COUNTS[-1]}³ generator, got {size}³."
        )
    order = nested_plane_order(size)
    provenance = build_provenance(
        weights,
        guidance,
        generation={
            "axis": AXIS,
            "counts": list(COUNTS),
            "domain": args.domain,
            "reference_seed": REFERENCE_SEED,
            "generation_seed": GENERATION_SEED,
            "anchor_strength": anchor_strength,
            "selection": "nested farthest-point plane order",
            "margin": generator.default_margin,
        },
    )
    validate_output_paths(provenance, (OUTPUT, METADATA))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(REFERENCE_SEED, device)
    print("Generating controlled reference...", flush=True)
    reference = generator.generate(
        anchors=(),
        anchor_strength=0.0,
        guidance=guidance,
        domain=args.domain,
        margin=generator.default_margin,
    )
    reference_slices = reference.movedim(AXIS, 0)
    rows: list[dict[str, object]] = []
    for count in COUNTS:
        indices = tuple(sorted(order[:count]))
        anchors = tuple(
            PlaneAnchor(image=reference_slices[index], axis=AXIS, index=index)
            for index in indices
        )
        set_seed(GENERATION_SEED, device)
        print(f"Generating with {count:3d} nested planes...", flush=True)
        generated = generator.generate(
            anchors=anchors,
            anchor_strength=anchor_strength,
            guidance=guidance,
            domain=args.domain,
            margin=generator.default_margin,
        )
        accuracy = voxel_accuracy(generated, reference)
        rows.append(
            {
                "anchor_count": count,
                "coverage": count / size,
                "voxel_accuracy": accuracy,
                "indices": list(indices),
            }
        )
        print(f"  whole-volume agreement: {accuracy:.2%}", flush=True)

    render(rows, OUTPUT)
    METADATA.write_text(
        json.dumps(
            {
                **provenance,
                "metric": {
                    "name": "whole-volume voxel agreement",
                    "definition": (
                        "fraction of all generated voxels equal to the synthetic "
                        "reference at the same coordinate"
                    ),
                    "scope": "controlled same-model diagnostic, not experimental GT",
                },
                "results": rows,
                "output": file_record(OUTPUT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Figure  : {OUTPUT.resolve()}")
    print(f"Metadata: {METADATA.resolve()}")


def nested_plane_order(size: int) -> tuple[int, ...]:
    """Return a deterministic nested order with broad coverage at every prefix."""
    if size <= 0:
        raise ValueError("size must be positive.")
    selected: list[int] = []
    remaining = set(range(size))
    while remaining:
        if not selected:
            choice = size // 2
        else:
            choice = max(
                remaining,
                key=lambda index: (
                    min(abs(index - prior) for prior in selected),
                    -abs(index - (size - 1) / 2),
                    -index,
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def render(rows: list[dict[str, object]], output: Path) -> None:
    counts = [int(row["anchor_count"]) for row in rows]
    scores = [100.0 * float(row["voxel_accuracy"]) for row in rows]
    positions = list(range(len(counts)))
    figure, axis = plt.subplots(figsize=(8.2, 4.6), facecolor="white")
    axis.plot(
        positions,
        scores,
        color="#276FBF",
        marker="o",
        linewidth=2.2,
        markersize=6,
    )
    for position, score in zip(positions, scores, strict=True):
        axis.annotate(
            f"{score:.1f}%",
            (position, score),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#1f2933",
        )
    axis.set_xticks(positions, [str(count) for count in counts])
    axis.set_xlabel("Nested axis-0 anchor planes")
    axis.set_ylabel("Whole-volume voxel agreement (%)")
    axis.set_title("(d) Controlled 3D recovery as reference sections are added")
    axis.set_ylim(0.0, 105.0)
    axis.grid(axis="y", color="#d9dee5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
