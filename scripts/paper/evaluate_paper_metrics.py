import argparse
import csv
import json
import sys
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from make_anchor_asset import AXIS, load_center_roi
from make_assets import CROP_SIZE, SAMPLE_PATH
from provenance import (
    build_provenance,
    describe_files,
    sha256_file,
    validate_manifest,
    validate_output_paths,
)

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.evaluate import (
    kid_score,
    make_kid_metric,
    phase_fraction,
    tortuosity,
    voxel_accuracy,
)
from src.scale import DEFAULT_SCALE_OVERLAP, ScaledGenerator

REFERENCE_PATH = PROJECT_ROOT / "scripts" / "gt_128.tiff"
TEMP_DIR = PROJECT_ROOT / "temp"
VOLUME_DIR = TEMP_DIR / "paper_metrics"
RAW_CSV = TEMP_DIR / "paper_metrics_raw.csv"
SUMMARY_CSV = TEMP_DIR / "paper_metrics_summary.csv"
MANIFEST_PATH = TEMP_DIR / "paper_metrics_manifest.json"

SEEDS = (0, 1, 2, 3)
ANCHOR_COUNTS = (32, 64, 96, 128)
PORE_PHASE = 0
REAL_CROP_COUNT = 64
REAL_REFERENCE_SEED = 10_000
REAL_EVALUATION_SEEDS = (20_000, 20_001, 20_002, 20_003)
KID_SUBSETS = 100
KID_SUBSET_SIZE = 50
TAUFACTOR_CONVERGENCE = 1e-3
SCALE_BLOCKS = (3, 3, 3)
SCALE_OVERLAP = DEFAULT_SCALE_OVERLAP
SEAM_EXCLUSION_RADIUS = 4

CONDITIONS = (
    "Real 2D crops",
    "3D",
    "3D (phase-fraction conditioned)",
    "3D (anchored, 25%)",
    "3D (anchored, 50%)",
    "3D (anchored, 75%)",
    "3D (anchored, 100%)",
    "3D (scale-up)",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse cached TIFF volumes and recompute only their metrics",
    )
    args = parser.parse_args()

    reference_digest = sha256_file(REFERENCE_PATH)
    training_image_digest = sha256_file(SAMPLE_PATH)
    reference = load_volume(REFERENCE_PATH)
    patch_size = reference.shape[0]
    if reference.shape != (patch_size,) * 3:
        raise ValueError("GT reference must be cubic.")
    real_reference = sample_real_crops(REAL_REFERENCE_SEED, patch_size)
    if sha256_file(REFERENCE_PATH) != reference_digest:
        raise ValueError("paper reference changed while it was being loaded.")
    if sha256_file(SAMPLE_PATH) != training_image_digest:
        raise ValueError("training image changed while it was being loaded.")
    target_porosity = phase_fraction(reference, PORE_PHASE)

    VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    weights = args.weight.resolve()
    generation = {
        "conditions": list(CONDITIONS[1:]),
        "domain": args.domain,
        "volume_seeds": list(SEEDS),
        "axis": AXIS,
        "anchor_counts": list(ANCHOR_COUNTS),
        "scale_geometry": "fixed_blocks_inward_margins",
        "scale_blocks": list(SCALE_BLOCKS),
        "scale_overlap": SCALE_OVERLAP,
        "scale_output_shape": list(scale_output_shape(patch_size)),
        "reference_shape": list(reference.shape),
        "phase_fraction_target": [target_porosity, 1.0 - target_porosity],
    }
    provenance = build_provenance(
        weights,
        args.guidance_scale,
        generation=generation,
        reference=REFERENCE_PATH,
        additional_inputs={"training_image": SAMPLE_PATH},
    )
    if provenance["reference_sha256"] != reference_digest:
        raise ValueError("paper reference changed before provenance was recorded.")
    training_record = provenance["additional_inputs"]["training_image"]
    if training_record["sha256"] != training_image_digest:
        raise ValueError("training image changed before provenance was recorded.")
    cached_paths = [
        volume_path(condition, seed) for condition in CONDITIONS[1:] for seed in SEEDS
    ]
    validate_output_paths(
        provenance,
        (*cached_paths, RAW_CSV, SUMMARY_CSV, MANIFEST_PATH),
    )
    if args.reuse:
        validate_manifest(
            MANIFEST_PATH,
            provenance,
            label="paper metrics reuse",
            cached_paths=cached_paths,
        )
    else:
        generate_volumes(
            reference,
            target_porosity,
            weights,
            args.guidance_scale,
            args.domain,
            provenance,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_rows, kid_scores = compute_kid_scores(
        real_reference,
        patch_size,
        device,
    )

    # KID's Inception model is released before TauFactor allocates its solver fields.
    reference_tortuosity = tortuosity(
        reference,
        phase=PORE_PHASE,
        axis=AXIS,
        device=device,
        convergence=TAUFACTOR_CONVERGENCE,
    )
    for condition in CONDITIONS[1:]:
        for seed in SEEDS:
            volume = load_volume(volume_path(condition, seed))
            accuracy = None
            seam_drop = None
            if condition.startswith("3D (anchored"):
                accuracy = voxel_accuracy(volume, reference)
            if condition == "3D (scale-up)":
                seams = scale_seams(volume.shape, patch_size, SCALE_OVERLAP)
                seam_drop = seam_connectivity_drop(volume, seams)
            kid_mean, kid_subset_std = kid_scores[(condition, seed)]
            raw_rows.append(
                make_row(
                    condition=condition,
                    seed=seed,
                    guidance_scale=args.guidance_scale,
                    kid=kid_mean,
                    kid_subset_std=kid_subset_std,
                    porosity_value=phase_fraction(volume, PORE_PHASE),
                    tortuosity_value=tortuosity(
                        volume,
                        phase=PORE_PHASE,
                        axis=AXIS,
                        device=device,
                        convergence=TAUFACTOR_CONVERGENCE,
                    ),
                    voxel_accuracy_value=accuracy,
                    seam_connectivity_drop_value=seam_drop,
                )
            )
            print_row(raw_rows[-1])

    write_csv(raw_rows, RAW_CSV)
    summary_rows = summarize(
        raw_rows,
        reference_porosity=target_porosity,
        reference_tortuosity=reference_tortuosity,
        guidance_scale=args.guidance_scale,
    )
    write_csv(summary_rows, SUMMARY_CSV)
    write_manifest(
        provenance=provenance,
        cached_paths=cached_paths,
        derived_outputs=(RAW_CSV, SUMMARY_CSV),
        reference=reference,
        reference_porosity=target_porosity,
        reference_tortuosity=reference_tortuosity,
        output=MANIFEST_PATH,
    )
    print(f"Raw metrics : {RAW_CSV.resolve()}")
    print(f"Summary     : {SUMMARY_CSV.resolve()}")
    print(f"Manifest    : {MANIFEST_PATH.resolve()}")


def generate_volumes(
    reference: np.ndarray,
    target_porosity: float,
    weights: Path,
    guidance_scale: float,
    domain: int,
    provenance: dict[str, object],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(weights, device=device)
    patch_size = generator.patch_size
    if reference.shape != (patch_size,) * 3:
        raise ValueError("GT reference must match the generator patch size.")
    reference_tensor = torch.from_numpy(reference.copy()).to(torch.long)
    reference_slices = reference_tensor.movedim(AXIS, 0)
    center_anchor = load_center_roi(patch_size)
    center_index = patch_size // 2
    anchor_map = {
        f"3D (anchored, {coverage}%)": count
        for coverage, count in zip((25, 50, 75, 100), ANCHOR_COUNTS, strict=True)
    }

    print("\nPaper quantitative generation")
    print("-----------------------------")
    print(f"Weights : {weights.resolve()}")
    print(f"Device  : {device}")
    print(f"Guidance: {guidance_scale}")
    for condition in CONDITIONS[1:]:
        for seed in SEEDS:
            set_seed(seed, device)
            print(f"Generating {condition}, seed={seed}...", flush=True)
            if condition == "3D":
                volume = generator.generate(
                    vf=None,
                    guidance_scale=guidance_scale,
                    domain=domain,
                )
            elif condition == "3D (phase-fraction conditioned)":
                volume = generator.generate(
                    vf=(target_porosity, 1.0 - target_porosity),
                    guidance_scale=guidance_scale,
                    domain=domain,
                )
            elif condition in anchor_map:
                count = anchor_map[condition]
                anchors = tuple(
                    PlaneAnchor(
                        image=reference_slices[index],
                        axis=AXIS,
                        index=index,
                    )
                    for index in select_indices(patch_size, count)
                )
                volume = generator.generate(
                    anchors=anchors,
                    guidance_scale=guidance_scale,
                    domain=domain,
                )
            elif condition == "3D (scale-up)":
                base = generator.generate(
                    anchors=(
                        PlaneAnchor(
                            image=center_anchor,
                            axis=AXIS,
                            index=center_index,
                        ),
                    ),
                    guidance_scale=guidance_scale,
                    domain=domain,
                )
                scaled = ScaledGenerator(generator)
                volume = scaled.generate(
                    blocks=SCALE_BLOCKS,
                    overlap=SCALE_OVERLAP,
                    base=base,
                    progress=False,
                    guidance_scale=guidance_scale,
                    domain=domain,
                )
                expected_shape = scaled.shape_from_blocks(SCALE_BLOCKS, SCALE_OVERLAP)
                if tuple(volume.shape) != expected_shape:
                    raise RuntimeError("scale-up output does not match its block plan.")
            else:
                raise RuntimeError(f"unsupported condition: {condition}")
            path = volume_path(condition, seed)
            tifffile.imwrite(path, volume.to(torch.uint8).numpy())
            print(f"Saved      {path.resolve()}")

    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()


def sample_real_crops(seed: int, output_size: int) -> np.ndarray:
    with Image.open(SAMPLE_PATH) as image:
        labels = np.asarray(image).copy()
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("training image must be a 2D uint8 phase map.")
    height, width = labels.shape
    if height < CROP_SIZE or width < CROP_SIZE:
        raise ValueError("training image is smaller than the crop size.")
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(REAL_CROP_COUNT):
        top = int(rng.integers(0, height - CROP_SIZE + 1))
        left = int(rng.integers(0, width - CROP_SIZE + 1))
        crop = torch.from_numpy(
            labels[top : top + CROP_SIZE, left : left + CROP_SIZE].copy()
        )[None, None].to(torch.float32)
        resized = F.interpolate(
            crop,
            size=(output_size, output_size),
            mode="nearest",
        )[0, 0]
        crops.append(resized.to(torch.uint8).numpy())
    return np.stack(crops)


def compute_kid_scores(
    real: np.ndarray,
    patch_size: int,
    device: torch.device,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, int], tuple[float, float]],
]:
    metric = make_kid_metric(real, device)
    real_rows: list[dict[str, object]] = []
    generated_scores: dict[tuple[str, int], tuple[float, float]] = {}
    try:
        for seed in REAL_EVALUATION_SEEDS:
            crops = sample_real_crops(seed, patch_size)
            kid_mean, kid_subset_std = kid_score(metric, crops, device)
            real_rows.append(
                make_row(
                    condition="Real 2D crops",
                    seed=seed,
                    kid=kid_mean,
                    kid_subset_std=kid_subset_std,
                    porosity_value=phase_fraction(crops, PORE_PHASE),
                )
            )

        for condition in CONDITIONS[1:]:
            for seed in SEEDS:
                volume = load_volume(volume_path(condition, seed))
                slices = select_metric_slices(
                    volume,
                    axis=AXIS,
                    count=REAL_CROP_COUNT,
                    output_size=patch_size,
                    seed=seed,
                )
                generated_scores[(condition, seed)] = kid_score(
                    metric,
                    slices,
                    device,
                )
    finally:
        del metric
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return real_rows, generated_scores


def select_metric_slices(
    volume: np.ndarray,
    axis: int,
    count: int,
    output_size: int,
    seed: int,
) -> np.ndarray:
    slices = np.moveaxis(volume, axis, 0)
    if count > slices.shape[0]:
        raise ValueError("requested more KID sections than the volume contains.")
    rng = np.random.default_rng(30_000 + seed)
    indices = np.sort(rng.choice(slices.shape[0], size=count, replace=False))
    selected = slices[np.asarray(indices)]
    if selected.shape[1:] == (output_size, output_size):
        return selected
    if selected.shape[1] < output_size or selected.shape[2] < output_size:
        raise ValueError("generated sections are smaller than the KID field of view.")
    crops = []
    for section in selected:
        top = int(rng.integers(0, section.shape[0] - output_size + 1))
        left = int(rng.integers(0, section.shape[1] - output_size + 1))
        crops.append(section[top : top + output_size, left : left + output_size])
    return np.stack(crops)


def select_indices(size: int, count: int) -> tuple[int, ...]:
    if count < 1 or count > size:
        raise ValueError("count must be between one and the axis size.")
    return tuple((2 * index + 1) * size // (2 * count) for index in range(count))


def scale_output_shape(patch_size: int) -> tuple[int, int, int]:
    stride = patch_size - 2 * SCALE_OVERLAP
    return tuple(patch_size + (count - 1) * stride for count in SCALE_BLOCKS)


def scale_seams(
    shape: tuple[int, ...],
    patch_size: int,
    overlap: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if 2 * overlap >= patch_size:
        raise ValueError("twice overlap must be smaller than patch_size.")
    stride = patch_size - 2 * overlap
    starts = tuple(
        ScaledGenerator.axis_starts(size, patch_size, stride) for size in shape
    )
    return tuple(  # type: ignore[return-value]
        tuple((left + patch_size + right) // 2 for left, right in pairwise(axis))
        for axis in starts
    )


def seam_connectivity_drop(
    volume: np.ndarray,
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> float:
    """Interior pore continuation minus continuation at exact tile seams."""
    drops = []
    for axis, positions in enumerate(seams):
        seam_pairs = {position - 1 for position in positions}
        excluded = {
            index
            for seam_index in seam_pairs
            for index in range(
                max(0, seam_index - SEAM_EXCLUSION_RADIUS),
                min(volume.shape[axis] - 1, seam_index + SEAM_EXCLUSION_RADIUS + 1),
            )
        }
        seam_same = seam_total = interior_same = interior_total = 0
        for index in range(volume.shape[axis] - 1):
            previous = np.take(volume, index, axis=axis) == PORE_PHASE
            current = np.take(volume, index + 1, axis=axis) == PORE_PHASE
            total = int(previous.sum())
            same = int(np.logical_and(previous, current).sum())
            if index in seam_pairs:
                seam_same += same
                seam_total += total
            elif index not in excluded:
                interior_same += same
                interior_total += total
        if seam_total == 0 or interior_total == 0:
            raise ValueError(
                "cannot calculate seam connectivity with an empty pore set."
            )
        seam_connectivity = seam_same / seam_total
        interior_connectivity = interior_same / interior_total
        drops.append(interior_connectivity - seam_connectivity)
    return float(np.mean(drops))


def make_row(
    condition: str,
    seed: int,
    guidance_scale: float | None = None,
    kid: float | None = None,
    kid_subset_std: float | None = None,
    porosity_value: float | None = None,
    tortuosity_value: float | None = None,
    voxel_accuracy_value: float | None = None,
    seam_connectivity_drop_value: float | None = None,
) -> dict[str, object]:
    return {
        "condition": condition,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "kid": kid,
        "kid_subset_std": kid_subset_std,
        "porosity": porosity_value,
        "tortuosity_axis0": tortuosity_value,
        "voxel_accuracy": voxel_accuracy_value,
        "seam_connectivity_drop": seam_connectivity_drop_value,
    }


def summarize(
    raw_rows: list[dict[str, object]],
    reference_porosity: float,
    reference_tortuosity: float,
    guidance_scale: float,
) -> list[dict[str, object]]:
    summary = [
        {
            "condition": "GT reference volume",
            "guidance_scale": None,
            "samples": 1,
            "kid_mean": None,
            "kid_std": None,
            "porosity_mean": reference_porosity,
            "porosity_std": None,
            "tortuosity_mean": reference_tortuosity,
            "tortuosity_std": None,
            "voxel_accuracy_mean": None,
            "voxel_accuracy_std": None,
            "seam_connectivity_drop_mean": None,
            "seam_connectivity_drop_std": None,
        }
    ]
    for condition in CONDITIONS:
        rows = [row for row in raw_rows if row["condition"] == condition]
        result: dict[str, object] = {
            "condition": condition,
            "guidance_scale": (
                None if condition == "Real 2D crops" else guidance_scale
            ),
            "samples": len(rows),
        }
        kid_values = np.asarray(
            [row["kid"] for row in rows if row["kid"] is not None],
            dtype=float,
        )
        kid_subset_stds = np.asarray(
            [
                row["kid_subset_std"]
                for row in rows
                if row["kid_subset_std"] is not None
            ],
            dtype=float,
        )
        result["kid_mean"] = float(kid_values.mean()) if kid_values.size else None
        result["kid_std"] = (
            float(kid_subset_stds.mean()) if kid_subset_stds.size else None
        )
        for metric in (
            "porosity",
            "tortuosity_axis0",
            "voxel_accuracy",
            "seam_connectivity_drop",
        ):
            values = np.asarray(
                [row[metric] for row in rows if row[metric] is not None],
                dtype=float,
            )
            name = "tortuosity" if metric == "tortuosity_axis0" else metric
            result[f"{name}_mean"] = float(values.mean()) if values.size else None
            result[f"{name}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else None
            )
        summary.append(result)
    return summary


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    provenance: dict[str, object],
    cached_paths: list[Path],
    derived_outputs: tuple[Path, ...],
    reference: np.ndarray,
    reference_porosity: float,
    reference_tortuosity: float,
    output: Path,
) -> None:
    data = {
        **provenance,
        "cached_outputs": describe_files(cached_paths),
        "outputs": describe_files(derived_outputs),
        "reference_shape": list(reference.shape),
        "training_image": str(SAMPLE_PATH.resolve()),
        "pore_phase": PORE_PHASE,
        "axis": AXIS,
        "volume_seeds": list(SEEDS),
        "real_reference_seed": REAL_REFERENCE_SEED,
        "real_evaluation_seeds": list(REAL_EVALUATION_SEEDS),
        "real_crops_per_sample": REAL_CROP_COUNT,
        "crop_size": CROP_SIZE,
        "model_section_size": reference.shape[0],
        "phase_fraction_target": [
            reference_porosity,
            1.0 - reference_porosity,
        ],
        "anchor_counts": list(ANCHOR_COUNTS),
        "scale_geometry": "fixed_blocks_inward_margins",
        "scale_blocks": list(SCALE_BLOCKS),
        "scale_overlap": SCALE_OVERLAP,
        "scale_output_shape": list(scale_output_shape(reference.shape[0])),
        "scale_seams": [
            list(axis)
            for axis in scale_seams(
                scale_output_shape(reference.shape[0]),
                reference.shape[0],
                SCALE_OVERLAP,
            )
        ],
        "reference_porosity": reference_porosity,
        "reference_tortuosity_axis0": reference_tortuosity,
        "kid": {
            "implementation": "torchmetrics.image.kid.KernelInceptionDistance",
            "torchmetrics_version": version("torchmetrics"),
            "feature_dimensions": 2048,
            "subsets": KID_SUBSETS,
            "subset_size": KID_SUBSET_SIZE,
            "real_images": REAL_CROP_COUNT,
            "generated_axis0_slices": REAL_CROP_COUNT,
            "field_of_view": [reference.shape[0], reference.shape[0]],
            "generated_section_sampling": (
                f"{REAL_CROP_COUNT} axis-0 sections sampled without replacement "
                "with seed 30000 + volume seed"
            ),
            "scale_up_crop": (
                f"one deterministic {reference.shape[0]} x {reference.shape[0]} crop "
                "from each sampled scale-up section"
            ),
            "mean_aggregation": "arithmetic mean of four per-seed KID estimates",
            "reported_std": (
                "mean standard deviation across 100 KID subsets over four seeds"
            ),
        },
        "tortuosity": {
            "implementation": "TauFactor steady-state diffusion Solver",
            "taufactor_version": version("taufactor"),
            "conductive_phase": PORE_PHASE,
            "axis": AXIS,
            "convergence_criterion": TAUFACTOR_CONVERGENCE,
        },
        "seam_connectivity_drop": {
            "definition": "mean_axis(interior pore continuation - exact seam pore continuation)",
            "interior_exclusion_radius": SEAM_EXCLUSION_RADIUS,
        },
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def volume_path(condition: str, seed: int) -> Path:
    slug = (
        condition.lower()
        .replace("3d ", "")
        .replace("3d", "unconditioned")
        .replace("phase-fraction ", "vf_")
        .replace("anchored, ", "anchor_")
        .replace("scale-up", "scale_up")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "")
        .replace(" ", "_")
    )
    return VOLUME_DIR / f"{slug}_seed_{seed}.tiff"


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(tifffile.imread(path))
    if volume.ndim != 3 or volume.dtype != np.uint8 or volume.size == 0:
        raise ValueError(f"volume must be a non-empty 3D uint8 array: {path}")
    if int(volume.max()) > 1:
        raise ValueError(f"volume contains a phase outside 0 and 1: {path}")
    return volume


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def print_row(row: dict[str, object]) -> None:
    values = []
    for key in (
        "kid",
        "porosity",
        "tortuosity_axis0",
        "voxel_accuracy",
        "seam_connectivity_drop",
    ):
        value = row[key]
        values.append(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}=-")
    print(f"{row['condition']} seed={row['seed']}: " + " ".join(values))


if __name__ == "__main__":
    main()
