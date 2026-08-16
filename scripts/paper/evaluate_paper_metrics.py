import argparse
import csv
import json
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from evaluation_io import (
    load_binary_volume as load_volume,
)
from evaluation_io import (
    start_generation_timer,
    stop_generation_timer,
    write_csv,
)
from make_anchor_asset import AXIS, load_center_roi
from make_assets import CROP_SIZE, OUTPUT_DIR, ROI_COLOR, SAMPLE_PATH
from provenance import (
    build_provenance,
    describe_files,
    sha256_file,
    validate_manifest,
    validate_output_paths,
)

from src.anchor import PlaneAnchor
from src.build import load_generator
from src.config import load_generation_settings
from src.evaluate import (
    fid_score,
    make_fid_metric,
    percolating_fractions,
    phase_fraction,
    tortuosity,
    voxel_accuracy,
)
from src.generate import Generator
from src.scale import ScaledGenerator

REFERENCE_PATH = PROJECT_ROOT / "scripts" / "gt.tiff"
TEMP_DIR = PROJECT_ROOT / "temp"
VOLUME_DIR = TEMP_DIR / "paper_metrics"
RAW_CSV = TEMP_DIR / "paper_metrics_raw.csv"
SUMMARY_CSV = TEMP_DIR / "paper_metrics_summary.csv"
MANIFEST_PATH = TEMP_DIR / "paper_metrics_manifest.json"
FIGURE_PATH = OUTPUT_DIR / "08-paper-metrics.png"

SEEDS = (0, 1, 2, 3)
ANCHOR_COUNTS = (32, 64, 96, 128)
PORE_PHASE = 0
REAL_CROP_COUNT = 64
REAL_REFERENCE_SEED = 10_000
REAL_EVALUATION_SEEDS = (20_000, 20_001, 20_002, 20_003)
TAUFACTOR_CONVERGENCE = 1e-3
SCALE_BLOCKS = (3, 3, 3)
SCALE_MARGIN = 0

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
    parser.add_argument("--guidance", type=float)
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
    generation_settings = load_generation_settings()
    guidance = generation_settings.guidance if args.guidance is None else args.guidance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(weights, device=device)
    margin = generator.default_margin
    generation = {
        "conditions": list(CONDITIONS[1:]),
        "domain": args.domain,
        "volume_seeds": list(SEEDS),
        "axis": AXIS,
        "anchor_counts": list(ANCHOR_COUNTS),
        "scale_geometry": "fixed_blocks_inward_margins",
        "scale_blocks": list(SCALE_BLOCKS),
        "scale_overlap": generation_settings.overlap,
        "scale_margin": SCALE_MARGIN,
        "margin": margin,
        "scale_output_shape": list(
            scale_output_shape(patch_size, generation_settings.overlap)
        ),
        "reference_shape": list(reference.shape),
        "phase_fraction_target": [target_porosity, 1.0 - target_porosity],
    }
    provenance = build_provenance(
        weights,
        guidance,
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
        (*cached_paths, RAW_CSV, SUMMARY_CSV, MANIFEST_PATH, FIGURE_PATH),
    )
    if args.reuse:
        validate_manifest(
            MANIFEST_PATH,
            provenance,
            label="paper metrics reuse",
            cached_paths=cached_paths,
        )
        generation_times = load_generation_times(RAW_CSV)
    else:
        generation_times = generate_volumes(
            reference,
            target_porosity,
            weights,
            guidance,
            args.domain,
            generation_settings.overlap,
            margin,
            generator,
        )

    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_rows, reference_fid, fid_scores = compute_fid_scores(
        real_reference,
        reference,
        patch_size,
        device,
    )

    reference_tortuosity = tortuosity(
        reference,
        phase=PORE_PHASE,
        axis=AXIS,
        device=device,
        convergence=TAUFACTOR_CONVERGENCE,
    )
    reference_percolation = mean_percolation(reference)
    for condition in CONDITIONS[1:]:
        for seed in SEEDS:
            volume = load_volume(volume_path(condition, seed))
            accuracy = None
            if condition.startswith("3D (anchored"):
                accuracy = voxel_accuracy(volume, reference)
            raw_rows.append(
                make_row(
                    condition=condition,
                    seed=seed,
                    guidance=guidance,
                    fid=fid_scores[(condition, seed)],
                    porosity_value=phase_fraction(volume, PORE_PHASE),
                    tortuosity_value=tortuosity(
                        volume,
                        phase=PORE_PHASE,
                        axis=AXIS,
                        device=device,
                        convergence=TAUFACTOR_CONVERGENCE,
                    ),
                    voxel_accuracy_value=accuracy,
                    percolation_value=mean_percolation(volume),
                    generation_seconds=generation_times[(condition, seed)],
                )
            )
            print_row(raw_rows[-1])

    write_csv(raw_rows, RAW_CSV)
    summary_rows = summarize(
        raw_rows,
        reference_fid=reference_fid,
        reference_porosity=target_porosity,
        reference_tortuosity=reference_tortuosity,
        reference_percolation=reference_percolation,
        guidance=guidance,
    )
    write_csv(summary_rows, SUMMARY_CSV)
    render_summary_chart(summary_rows, FIGURE_PATH)
    write_manifest(
        provenance=provenance,
        cached_paths=cached_paths,
        derived_outputs=(RAW_CSV, SUMMARY_CSV, FIGURE_PATH),
        reference=reference,
        reference_fid=reference_fid,
        reference_porosity=target_porosity,
        reference_tortuosity=reference_tortuosity,
        reference_percolation=reference_percolation,
        scale_overlap=generation_settings.overlap,
        scale_margin=SCALE_MARGIN,
        output=MANIFEST_PATH,
    )
    print(f"Raw metrics : {RAW_CSV.resolve()}")
    print(f"Summary     : {SUMMARY_CSV.resolve()}")
    print(f"Figure      : {FIGURE_PATH.resolve()}")
    print(f"Manifest    : {MANIFEST_PATH.resolve()}")


def generate_volumes(
    reference: np.ndarray,
    target_porosity: float,
    weights: Path,
    guidance: float,
    domain: int,
    scale_overlap: int,
    margin: int,
    generator: Generator,
) -> dict[tuple[str, int], float]:
    device = generator.device
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
    print(f"Guidance: {guidance}")
    elapsed_times = {}
    for condition in CONDITIONS[1:]:
        for seed in SEEDS:
            set_seed(seed, device)
            print(f"Generating {condition}, seed={seed}...", flush=True)
            start = start_generation_timer(device)
            if condition == "3D":
                volume = generator.generate(
                    vf=None,
                    guidance=guidance,
                    domain=domain,
                    margin=margin,
                )
            elif condition == "3D (phase-fraction conditioned)":
                volume = generator.generate(
                    vf=(target_porosity, 1.0 - target_porosity),
                    guidance=guidance,
                    domain=domain,
                    margin=margin,
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
                    guidance=guidance,
                    domain=domain,
                    margin=margin,
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
                    guidance=guidance,
                    domain=domain,
                    margin=margin,
                )
                scaled = ScaledGenerator(generator)
                volume = scaled.generate(
                    blocks=SCALE_BLOCKS,
                    overlap=scale_overlap,
                    margin=SCALE_MARGIN,
                    base=base,
                    progress=False,
                    guidance=guidance,
                    domain=domain,
                )
                expected_shape = scaled.shape_from_blocks(SCALE_BLOCKS, scale_overlap)
                if tuple(volume.shape) != expected_shape:
                    raise RuntimeError("scale-up output does not match its block plan.")
            else:
                raise RuntimeError(f"unsupported condition: {condition}")
            elapsed = stop_generation_timer(device, start)
            elapsed_times[(condition, seed)] = elapsed
            path = volume_path(condition, seed)
            tifffile.imwrite(path, volume.to(torch.uint8).numpy())
            print(f"Saved      {path.resolve()} ({elapsed:.3f} s)")

    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed_times


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


def compute_fid_scores(
    real: np.ndarray,
    reference: np.ndarray,
    patch_size: int,
    device: torch.device,
) -> tuple[
    list[dict[str, object]],
    float,
    dict[tuple[str, int], float],
]:
    metric = make_fid_metric(real, device)
    real_rows: list[dict[str, object]] = []
    generated_scores: dict[tuple[str, int], float] = {}
    try:
        for seed in REAL_EVALUATION_SEEDS:
            crops = sample_real_crops(seed, patch_size)
            real_rows.append(
                make_row(
                    condition="Real 2D crops",
                    seed=seed,
                    fid=fid_score(metric, crops, device),
                    porosity_value=phase_fraction(crops, PORE_PHASE),
                )
            )

        reference_slices = select_metric_slices(
            reference,
            axis=AXIS,
            count=REAL_CROP_COUNT,
            output_size=patch_size,
            seed=REAL_REFERENCE_SEED,
        )
        reference_fid = fid_score(metric, reference_slices, device)

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
                generated_scores[(condition, seed)] = fid_score(metric, slices, device)
    finally:
        del metric
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return real_rows, reference_fid, generated_scores


def select_metric_slices(
    volume: np.ndarray,
    axis: int,
    count: int,
    output_size: int,
    seed: int,
) -> np.ndarray:
    slices = np.moveaxis(volume, axis, 0)
    if count > slices.shape[0]:
        raise ValueError("requested more FID sections than the volume contains.")
    rng = np.random.default_rng(30_000 + seed)
    indices = np.sort(rng.choice(slices.shape[0], size=count, replace=False))
    selected = slices[np.asarray(indices)]
    if selected.shape[1:] == (output_size, output_size):
        return selected
    if selected.shape[1] < output_size or selected.shape[2] < output_size:
        raise ValueError("generated sections are smaller than the FID field of view.")
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


def scale_output_shape(
    patch_size: int,
    overlap: int,
) -> tuple[int, int, int]:
    stride = patch_size - 2 * overlap
    return tuple(patch_size + (count - 1) * stride for count in SCALE_BLOCKS)


def mean_percolation(volume: np.ndarray) -> float:
    return float(np.mean(percolating_fractions(volume, PORE_PHASE)))


def make_row(
    condition: str,
    seed: int,
    guidance: float | None = None,
    fid: float | None = None,
    porosity_value: float | None = None,
    tortuosity_value: float | None = None,
    voxel_accuracy_value: float | None = None,
    percolation_value: float | None = None,
    generation_seconds: float | None = None,
) -> dict[str, object]:
    return {
        "condition": condition,
        "seed": seed,
        "guidance": guidance,
        "fid": fid,
        "porosity": porosity_value,
        "tortuosity_axis0": tortuosity_value,
        "voxel_accuracy": voxel_accuracy_value,
        "percolation": percolation_value,
        "generation_seconds": generation_seconds,
    }


def summarize(
    raw_rows: list[dict[str, object]],
    reference_fid: float,
    reference_porosity: float,
    reference_tortuosity: float,
    reference_percolation: float,
    guidance: float,
) -> list[dict[str, object]]:
    summary = [
        {
            "condition": "GT reference volume",
            "guidance": None,
            "samples": 1,
            "fid_mean": reference_fid,
            "fid_std": None,
            "porosity_mean": reference_porosity,
            "porosity_std": None,
            "tortuosity_mean": reference_tortuosity,
            "tortuosity_std": None,
            "voxel_accuracy_mean": None,
            "voxel_accuracy_std": None,
            "percolation_mean": reference_percolation,
            "percolation_std": None,
            "generation_seconds_mean": None,
            "generation_seconds_std": None,
        }
    ]
    for condition in CONDITIONS:
        rows = [row for row in raw_rows if row["condition"] == condition]
        result: dict[str, object] = {
            "condition": condition,
            "guidance": None if condition == "Real 2D crops" else guidance,
            "samples": len(rows),
        }
        for metric in (
            "fid",
            "porosity",
            "tortuosity_axis0",
            "voxel_accuracy",
            "percolation",
            "generation_seconds",
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


def load_generation_times(path: Path) -> dict[tuple[str, int], float]:
    if not path.is_file():
        raise FileNotFoundError(f"generation times were not found: {path}")
    values = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            condition = row["condition"]
            raw = row.get("generation_seconds")
            if condition in CONDITIONS[1:] and raw:
                values[(condition, int(row["seed"]))] = float(raw)
    expected = {(condition, seed) for condition in CONDITIONS[1:] for seed in SEEDS}
    if set(values) != expected:
        raise ValueError("cached generation times do not match the evaluation set.")
    return values


def render_summary_chart(
    rows: list[dict[str, object]],
    output: Path,
) -> None:
    labels = {
        "GT reference volume": "GT",
        "Real 2D crops": "Real crops",
        "3D": "3D",
        "3D (phase-fraction conditioned)": "VF conditioned",
        "3D (anchored, 25%)": "Anchored 25%",
        "3D (anchored, 50%)": "Anchored 50%",
        "3D (anchored, 75%)": "Anchored 75%",
        "3D (anchored, 100%)": "Anchored 100%",
        "3D (scale-up)": "Scale-up",
    }
    specs = (
        ("fid", "FID ↓", 1.0, None),
        ("voxel_accuracy", "Voxel accuracy ↑ (%)", 100.0, None),
        ("tortuosity", "Tortuosity", 1.0, "tortuosity"),
        ("porosity", "Porosity, phase 0 (%)", 100.0, "porosity"),
        ("percolation", "Percolation, phase 0 (%)", 100.0, "percolation"),
        ("generation_seconds", "Generation time ↓ (s)", 1.0, None),
    )
    references = {
        key: next(
            (
                float(row[f"{key}_mean"])
                for row in rows
                if row["condition"] == "GT reference volume"
                and row[f"{key}_mean"] is not None
            ),
            None,
        )
        for key in ("tortuosity", "porosity", "percolation")
    }
    figure, panels = plt.subplots(2, 3, figsize=(15, 10), facecolor="white")
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.07,
        top=0.96,
        hspace=0.32,
        wspace=0.38,
    )
    for panel, letter, (key, title, scale, reference_key) in zip(
        panels.ravel(),
        "abcdef",
        specs,
        strict=True,
    ):
        selected = [row for row in rows if row[f"{key}_mean"] is not None]
        names = [labels[str(row["condition"])] for row in selected]
        means = [scale * float(row[f"{key}_mean"]) for row in selected]
        errors = [
            0.0 if row[f"{key}_std"] is None else scale * float(row[f"{key}_std"])
            for row in selected
        ]
        colors = [
            "#64748b"
            if row["condition"] == "GT reference volume"
            else "#0ea5e9"
            if row["condition"] == "Real 2D crops"
            else ROI_COLOR
            for row in selected
        ]
        bars = panel.barh(
            names,
            means,
            xerr=errors,
            color=colors,
            alpha=0.9,
            capsize=3,
        )
        panel.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in means],
            padding=3,
            fontsize=8,
        )
        if reference_key is not None and references[reference_key] is not None:
            reference = scale * float(references[reference_key])
            panel.axvline(
                reference,
                color="#334155",
                linewidth=1.2,
                linestyle="--",
                label=f"GT: {reference:.2f}",
            )
            panel.legend(frameon=False, fontsize=8, loc="best")
        panel.set_title(f"({letter}) {title}", loc="left", fontsize=11)
        panel.invert_yaxis()
        panel.grid(axis="x", color="#cbd5e1", linewidth=0.7, alpha=0.65)
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(labelsize=8, colors="#475569")
        panel.margins(x=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def write_manifest(
    provenance: dict[str, object],
    cached_paths: list[Path],
    derived_outputs: tuple[Path, ...],
    reference: np.ndarray,
    reference_fid: float,
    reference_porosity: float,
    reference_tortuosity: float,
    reference_percolation: float,
    scale_overlap: int,
    scale_margin: int,
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
        "scale_overlap": scale_overlap,
        "scale_margin": scale_margin,
        "scale_output_shape": list(
            scale_output_shape(reference.shape[0], scale_overlap)
        ),
        "reference_fid": reference_fid,
        "reference_porosity": reference_porosity,
        "reference_tortuosity_axis0": reference_tortuosity,
        "reference_percolation": reference_percolation,
        "fid": {
            "implementation": "torchmetrics.image.fid.FrechetInceptionDistance",
            "torchmetrics_version": version("torchmetrics"),
            "feature_dimensions": 2048,
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
            "aggregation": "mean of four per-seed FID estimates",
            "reported_std": "between-seed sample standard deviation",
        },
        "tortuosity": {
            "implementation": "TauFactor steady-state diffusion Solver",
            "taufactor_version": version("taufactor"),
            "conductive_phase": PORE_PHASE,
            "axis": AXIS,
            "convergence_criterion": TAUFACTOR_CONVERGENCE,
        },
        "percolation": {
            "phase": PORE_PHASE,
            "connectivity": 6,
            "periodic": False,
            "axes": [0, 1, 2],
            "value": (
                "mean fraction across axes of phase-0 voxels belonging to "
                "components spanning opposing faces"
            ),
        },
        "generation_time": {
            "clock": "time.perf_counter",
            "cuda_synchronized": True,
            "includes": "sampling, including anchored base sampling for scale-up",
            "excludes": "TIFF serialization and metric calculation",
            "reported_std": "between-seed sample standard deviation",
            "unit": "seconds",
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


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def print_row(row: dict[str, object]) -> None:
    values = []
    for key in (
        "fid",
        "porosity",
        "tortuosity_axis0",
        "voxel_accuracy",
        "percolation",
        "generation_seconds",
    ):
        value = row[key]
        values.append(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}=-")
    print(f"{row['condition']} seed={row['seed']}: " + " ".join(values))


if __name__ == "__main__":
    main()
