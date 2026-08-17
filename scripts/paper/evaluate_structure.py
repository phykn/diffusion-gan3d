"""Evaluate structural realism of direct and scaled generation."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from evaluation_io import (
    load_binary_volume,
    start_generation_timer,
    stop_generation_timer,
    write_csv,
)
from make_assets import CROP_SIZE, OUTPUT_DIR, SAMPLE_PATH
from provenance import (
    build_provenance,
    describe_files,
    validate_manifest,
    validate_output_paths,
)

from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import (
    fid_score,
    make_fid_metric,
    percolating_fractions,
    phase_fraction,
    tortuosity,
)
from src.scale import ScaledGenerator
from src.volume import save_volume

SEEDS = (0, 1, 2, 3)
REAL_REFERENCE_SEED = 10_000
REAL_EVALUATION_SEEDS = (20_000, 20_001, 20_002, 20_003)
REAL_CROP_COUNT = 64
PORE_PHASE = 0
TAUFACTOR_CONVERGENCE = 1e-3
SCALE_BLOCKS = (3, 3, 3)
SCALE_MARGIN = 0

TEMP_DIR = PROJECT_ROOT / "temp" / "paper_structure"
RAW_CSV = TEMP_DIR / "raw.csv"
MANIFEST = TEMP_DIR / "manifest.json"
RESULTS = OUTPUT_DIR / "structure-results.json"

DIRECT = "Direct 128^3"
SCALED = "Scale-up 352^3"
CONDITIONS = (DIRECT, SCALED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance", type=float)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse generated TIFF volumes and recompute their metrics",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weight.resolve()
    settings = load_generation_settings()
    guidance = settings.guidance if args.guidance is None else args.guidance
    generator = load_generator(weights, device=device)
    patch_size = generator.patch_size
    scaled_shape = ScaledGenerator(generator).shape_from_blocks(
        SCALE_BLOCKS, settings.overlap
    )
    generation = {
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "domain": args.domain,
        "direct_margin": generator.default_margin,
        "scale_blocks": list(SCALE_BLOCKS),
        "scale_overlap": settings.overlap,
        "scale_margin": SCALE_MARGIN,
        "scale_output_shape": list(scaled_shape),
    }
    provenance = build_provenance(
        weights,
        guidance,
        generation=generation,
        additional_inputs={"training_image": SAMPLE_PATH},
    )
    cached = tuple(
        volume_path(condition, seed) for condition in CONDITIONS for seed in SEEDS
    )
    validate_output_paths(provenance, (*cached, RAW_CSV, MANIFEST, RESULTS))
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reuse:
        validate_manifest(
            MANIFEST,
            provenance,
            label="structure evaluation reuse",
            cached_paths=cached,
        )
        generation_times = load_generation_times(RAW_CSV)
    else:
        generation_times = generate_volumes(
            generator,
            domain=args.domain,
            guidance=guidance,
            overlap=settings.overlap,
        )

    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_rows = evaluate(
        patch_size,
        device,
        guidance=guidance,
        generation_times=generation_times,
    )
    write_csv(raw_rows, RAW_CSV)
    summary = summarize(raw_rows)
    RESULTS.write_text(
        json.dumps(
            {
                **provenance,
                "metrics": {
                    "fid": "Inception-v3 FID on 64 axis-0 sections/crops",
                    "phase_0_fraction": "fraction of labels equal to phase 0",
                    "interface_density": (
                        "mean unlike-neighbor fraction over available axes"
                    ),
                    "tortuosity_axis0": "TauFactor phase-0 tortuosity",
                    "percolation": ("mean phase-0 spanning fraction over three axes"),
                },
                "raw": raw_rows,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    MANIFEST.write_text(
        json.dumps(
            {
                **provenance,
                "cached_outputs": describe_files(cached),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Raw metrics: {RAW_CSV.resolve()}")
    print(f"Results    : {RESULTS.resolve()}")


def generate_volumes(
    generator,
    *,
    domain: int,
    guidance: float,
    overlap: int,
) -> dict[tuple[str, int], float]:
    elapsed = {}
    scaled = ScaledGenerator(generator)
    for condition in CONDITIONS:
        for seed in SEEDS:
            set_seed(seed, generator.device)
            print(f"Generating {condition}, seed={seed}...", flush=True)
            start = start_generation_timer(generator.device)
            if condition == DIRECT:
                volume = generator.generate(
                    anchors=(),
                    anchor_strength=0.0,
                    guidance=guidance,
                    domain=domain,
                    margin=generator.default_margin,
                )
            else:
                volume = scaled.generate(
                    blocks=SCALE_BLOCKS,
                    overlap=overlap,
                    margin=SCALE_MARGIN,
                    progress=False,
                    guidance=guidance,
                    domain=domain,
                )
            elapsed[(condition, seed)] = stop_generation_timer(generator.device, start)
            save_volume(volume, volume_path(condition, seed))
    return elapsed


def evaluate(
    patch_size: int,
    device: torch.device,
    *,
    guidance: float,
    generation_times: dict[tuple[str, int], float],
) -> list[dict[str, object]]:
    real_reference = sample_real_crops(REAL_REFERENCE_SEED, patch_size)
    metric = make_fid_metric(real_reference, device)
    rows = []
    try:
        for seed in REAL_EVALUATION_SEEDS:
            crops = sample_real_crops(seed, patch_size)
            rows.append(
                make_row(
                    "Real 2D crops",
                    seed,
                    fid=fid_score(metric, crops, device),
                    phase_0_fraction=phase_fraction(crops, PORE_PHASE),
                    interface_density_value=interface_density(
                        crops, spatial_dimensions=2
                    ),
                )
            )
        for condition in CONDITIONS:
            for seed in SEEDS:
                volume = load_binary_volume(volume_path(condition, seed))
                sections = select_metric_slices(
                    volume,
                    axis=0,
                    count=REAL_CROP_COUNT,
                    output_size=patch_size,
                    seed=seed,
                )
                row = make_row(
                    condition,
                    seed,
                    guidance=guidance,
                    fid=fid_score(metric, sections, device),
                    phase_0_fraction=phase_fraction(volume, PORE_PHASE),
                    interface_density_value=interface_density(
                        volume, spatial_dimensions=3
                    ),
                    tortuosity_axis0=tortuosity(
                        volume,
                        phase=PORE_PHASE,
                        axis=0,
                        device=device,
                        convergence=TAUFACTOR_CONVERGENCE,
                    ),
                    percolation=float(
                        np.mean(percolating_fractions(volume, PORE_PHASE))
                    ),
                    generation_seconds=generation_times[(condition, seed)],
                )
                rows.append(row)
                print_result(row)
    finally:
        del metric
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def sample_real_crops(seed: int, output_size: int) -> np.ndarray:
    with Image.open(SAMPLE_PATH) as image:
        labels = np.asarray(image).copy()
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("training image must be a 2D uint8 phase map.")
    height, width = labels.shape
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(REAL_CROP_COUNT):
        top = int(rng.integers(0, height - CROP_SIZE + 1))
        left = int(rng.integers(0, width - CROP_SIZE + 1))
        crop = torch.from_numpy(
            labels[top : top + CROP_SIZE, left : left + CROP_SIZE].copy()
        )[None, None].to(torch.float32)
        crops.append(
            F.interpolate(crop, size=(output_size, output_size), mode="nearest")
            .to(torch.uint8)[0, 0]
            .numpy()
        )
    return np.stack(crops)


def select_metric_slices(
    volume: np.ndarray,
    *,
    axis: int,
    count: int,
    output_size: int,
    seed: int,
) -> np.ndarray:
    slices = np.moveaxis(volume, axis, 0)
    if count > slices.shape[0]:
        raise ValueError("requested more sections than the volume contains.")
    rng = np.random.default_rng(30_000 + seed)
    indices = np.sort(rng.choice(slices.shape[0], size=count, replace=False))
    selected = slices[np.asarray(indices)]
    if selected.shape[1:] == (output_size, output_size):
        return selected
    crops = []
    for section in selected:
        top = int(rng.integers(0, section.shape[0] - output_size + 1))
        left = int(rng.integers(0, section.shape[1] - output_size + 1))
        crops.append(section[top : top + output_size, left : left + output_size])
    return np.stack(crops)


def interface_density(labels: np.ndarray, *, spatial_dimensions: int) -> float:
    values = np.asarray(labels)
    if spatial_dimensions not in (2, 3) or values.ndim < spatial_dimensions:
        raise ValueError("spatial_dimensions must select two or three label axes.")
    spatial_axes = range(values.ndim - spatial_dimensions, values.ndim)
    changes = []
    for axis in spatial_axes:
        before = [slice(None)] * values.ndim
        after = [slice(None)] * values.ndim
        before[axis] = slice(None, -1)
        after[axis] = slice(1, None)
        changes.append(np.mean(values[tuple(before)] != values[tuple(after)]))
    return float(np.mean(changes))


def make_row(
    condition: str,
    seed: int,
    *,
    guidance: float | None = None,
    fid: float | None = None,
    phase_0_fraction: float | None = None,
    interface_density_value: float | None = None,
    tortuosity_axis0: float | None = None,
    percolation: float | None = None,
    generation_seconds: float | None = None,
) -> dict[str, object]:
    return {
        "condition": condition,
        "seed": seed,
        "guidance": guidance,
        "fid": fid,
        "phase_0_fraction": phase_0_fraction,
        "interface_density": interface_density_value,
        "tortuosity_axis0": tortuosity_axis0,
        "percolation": percolation,
        "generation_seconds": generation_seconds,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for condition in ("Real 2D crops", *CONDITIONS):
        selected = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {
            "condition": condition,
            "samples": len(selected),
        }
        for metric in (
            "fid",
            "phase_0_fraction",
            "interface_density",
            "tortuosity_axis0",
            "percolation",
            "generation_seconds",
        ):
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


def volume_path(condition: str, seed: int) -> Path:
    slug = "direct" if condition == DIRECT else "scale"
    return TEMP_DIR / f"{slug}_seed_{seed}.tiff"


def load_generation_times(path: Path) -> dict[tuple[str, int], float]:
    values = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            condition = row["condition"]
            raw = row.get("generation_seconds")
            if condition in CONDITIONS and raw:
                values[(condition, int(row["seed"]))] = float(raw)
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(values) != expected:
        raise ValueError("cached generation times do not match the evaluation set.")
    return values


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def print_result(row: dict[str, object]) -> None:
    print(
        f"{row['condition']}, seed={row['seed']}: "
        f"FID={row['fid']:.3f}, phase0={row['phase_0_fraction']:.4f}, "
        f"interface={row['interface_density']:.4f}, "
        f"tau={row['tortuosity_axis0']:.4f}, "
        f"percolation={row['percolation']:.4%}"
    )


if __name__ == "__main__":
    main()
