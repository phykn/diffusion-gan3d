import argparse
import csv
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from make_assets import OUTPUT_DIR, ROI_COLOR
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
    fid_score,
    make_fid_metric,
    percolating_fractions,
    phase_fraction,
    tortuosity,
    voxel_accuracy,
)

REFERENCE_PATH = PROJECT_ROOT / "scripts" / "gt_128.tiff"
TEMP_DIR = PROJECT_ROOT / "temp"
MANIFEST_PATH = TEMP_DIR / "anchor_sweep_manifest.json"
COUNTS = (128, 64, 32, 16, 8, 4, 2, 1, 0)
AXIS = 0
PORE_PHASE = 0
SEED = 0
TAUFACTOR_CONVERGENCE = 1e-3

FEATURE_DIMENSIONS = 2048


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse existing sweep TIFF files instead of regenerating them",
    )
    args = parser.parse_args()

    reference_digest = sha256_file(REFERENCE_PATH)
    reference = load_volume(REFERENCE_PATH)
    if sha256_file(REFERENCE_PATH) != reference_digest:
        raise ValueError("paper reference changed while it was being loaded.")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = args.weight.resolve()
    generation = {
        "seed": SEED,
        "domain": args.domain,
        "axis": AXIS,
        "anchor_counts": list(COUNTS),
        "reference_shape": list(reference.shape),
        "pore_phase": PORE_PHASE,
    }
    provenance = build_provenance(
        weights,
        args.guidance_scale,
        generation=generation,
        reference=REFERENCE_PATH,
    )
    if provenance["reference_sha256"] != reference_digest:
        raise ValueError("paper reference changed before provenance was recorded.")
    cached_paths = [volume_path(count) for count in COUNTS]
    csv_path = TEMP_DIR / "anchor_sweep_metrics.csv"
    figure_path = OUTPUT_DIR / "06-anchor-sweep-metrics.png"
    validate_output_paths(
        provenance,
        (*cached_paths, csv_path, MANIFEST_PATH, figure_path),
    )

    if args.reuse:
        reuse_keys = (
            "weights",
            "weight_sha256",
            "train_config",
            "train_config_sha256",
            "guidance_scale",
            "reference",
            "reference_sha256",
            "additional_inputs",
        )
        reuse_provenance = {key: provenance[key] for key in reuse_keys}
        cached_manifest = validate_manifest(
            MANIFEST_PATH,
            reuse_provenance,
            label="anchor sweep reuse",
            cached_paths=cached_paths,
        )
        cached_generation = cached_manifest.get("generation")
        if not isinstance(cached_generation, dict) or any(
            cached_generation.get(key) != value for key, value in generation.items()
        ):
            raise ValueError(
                "anchor sweep reuse manifest generation does not match current inputs."
            )
        generation_times = load_generation_times(csv_path)
    else:
        generation_times = generate_volumes(
            reference, weights, args.guidance_scale, args.domain
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_slices = get_slices(reference, AXIS)
    fid = make_fid_metric(
        reference_slices,
        device,
        feature=FEATURE_DIMENSIONS,
    )
    distribution_scores: dict[int, float] = {}
    for count in COUNTS:
        volume = load_volume(volume_path(count))
        slices = get_slices(volume, AXIS)
        distribution_scores[count] = fid_score(fid, slices, device)
    del fid
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference_porosity = phase_fraction(reference, PORE_PHASE)
    reference_tortuosity = tortuosity(
        reference,
        phase=PORE_PHASE,
        axis=AXIS,
        device=device,
        convergence=TAUFACTOR_CONVERGENCE,
    )
    reference_percolation = float(np.mean(percolating_fractions(reference, PORE_PHASE)))
    rows = []
    for count in COUNTS:
        path = volume_path(count)
        volume = load_volume(path)
        accuracy = voxel_accuracy(volume, reference)
        fid_value = distribution_scores[count]
        percolation = float(np.mean(percolating_fractions(volume, PORE_PHASE)))
        row = {
            "anchor_count": count,
            "coverage": count / reference.shape[AXIS],
            "guidance_scale": args.guidance_scale,
            "fid": fid_value,
            "porosity": phase_fraction(volume, PORE_PHASE),
            "tortuosity_axis0": tortuosity(
                volume,
                phase=PORE_PHASE,
                axis=AXIS,
                device=device,
                convergence=TAUFACTOR_CONVERGENCE,
            ),
            "voxel_accuracy": accuracy,
            "percolation": percolation,
            "generation_seconds": generation_times[count],
            "tiff": path.name,
        }
        rows.append(row)
        score = f"{accuracy:.4%}"
        print(
            f"count={count:2d} coverage={row['coverage']:.2%} "
            f"FID={row['fid']:.4f} "
            f"porosity={row['porosity']:.4f} "
            f"tortuosity={row['tortuosity_axis0']:.4f} accuracy={score} "
            f"percolation={row['percolation']:.4f} "
            f"time={row['generation_seconds']:.3f}s"
        )

    write_csv(rows, csv_path)
    render_chart(
        rows,
        reference_porosity=reference_porosity,
        reference_tortuosity=reference_tortuosity,
        reference_percolation=reference_percolation,
        output=figure_path,
    )
    write_manifest(
        provenance=provenance,
        cached_paths=cached_paths,
        derived_outputs=(csv_path, figure_path),
        reference=reference,
        reference_porosity=reference_porosity,
        reference_tortuosity=reference_tortuosity,
        reference_percolation=reference_percolation,
        output=MANIFEST_PATH,
    )
    print(f"Metrics : {csv_path.resolve()}")
    print(f"Manifest: {MANIFEST_PATH.resolve()}")
    print(f"Figure  : {figure_path.resolve()}")


def generate_volumes(
    reference: np.ndarray,
    weights: Path,
    guidance_scale: float,
    domain: int,
) -> dict[int, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(weights, device=device)
    if (
        generator.patch_size != reference.shape[0]
        or reference.shape != (generator.patch_size,) * 3
    ):
        raise ValueError("reference volume must match the generator patch size.")
    reference_tensor = torch.from_numpy(reference).to(torch.long)
    slices = reference_tensor.movedim(AXIS, 0)

    print("\nAnchor coverage sweep")
    print("---------------------")
    print(f"Weights : {weights.resolve()}")
    print(f"Reference: {REFERENCE_PATH.resolve()}")
    print(f"Device  : {device}")
    print(f"Guidance: {guidance_scale}")
    elapsed_times = {}
    for count in COUNTS:
        indices = select_indices(reference.shape[AXIS], count)
        anchors = tuple(
            PlaneAnchor(image=slices[index], axis=AXIS, index=index)
            for index in indices
        )
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        print(
            f"Status  : generating count={count:2d} "
            f"coverage={count / reference.shape[AXIS]:.2%}...",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        volume = generator.generate(
            anchors=anchors,
            guidance_scale=guidance_scale,
            domain=domain,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        elapsed_times[count] = elapsed
        path = volume_path(count)
        tifffile.imwrite(path, volume.numpy())
        print(f"Output  : {path.resolve()} ({elapsed:.3f} s)")
    return elapsed_times


def select_indices(size: int, count: int) -> tuple[int, ...]:
    """Match the coverage selection used by scripts/03_check_anchor.py."""
    if count == 0:
        return ()
    return tuple(((2 * index + 1) * size) // (2 * count) for index in range(count))


def volume_path(count: int) -> Path:
    return TEMP_DIR / f"anchor_axis0_count_{count:02d}.tiff"


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(tifffile.imread(path))
    if volume.ndim != 3 or volume.dtype != np.uint8 or volume.size == 0:
        raise ValueError(f"volume must be a non-empty 3D uint8 array: {path}")
    if int(volume.max()) > 1:
        raise ValueError(f"volume contains a phase outside 0 and 1: {path}")
    return volume


def get_slices(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(volume, axis, 0)


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_generation_times(path: Path) -> dict[int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"generation times were not found: {path}")
    values = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            raw = row.get("generation_seconds")
            if raw:
                values[int(row["anchor_count"])] = float(raw)
    if set(values) != set(COUNTS):
        raise ValueError("cached generation times do not match the anchor sweep.")
    return values


def write_manifest(
    provenance: dict[str, object],
    cached_paths: list[Path],
    derived_outputs: tuple[Path, ...],
    reference: np.ndarray,
    reference_porosity: float,
    reference_tortuosity: float,
    reference_percolation: float,
    output: Path,
) -> None:
    data = {
        **provenance,
        "cached_outputs": describe_files(cached_paths),
        "outputs": describe_files(derived_outputs),
        "reference_shape": list(reference.shape),
        "axis": AXIS,
        "pore_phase": PORE_PHASE,
        "seed": SEED,
        "anchor_counts": list(COUNTS),
        "reference_porosity": reference_porosity,
        "reference_tortuosity_axis0": reference_tortuosity,
        "reference_percolation": reference_percolation,
        "fid": {
            "reference_slices": reference.shape[0],
            "generated_slices": reference.shape[0],
            "implementation": "torchmetrics.image.fid.FrechetInceptionDistance",
            "package_version": version("torchmetrics"),
            "feature_extractor": "torch-fidelity Inception-v3 pool 3",
            "feature_dimensions": FEATURE_DIMENSIONS,
            "input": "uint8 RGB, phase 0 = 0 and phase 1 = 255",
            "axis": AXIS,
        },
        "pore_percolation": {
            "phase": PORE_PHASE,
            "connectivity": 6,
            "adjacency": "face-sharing voxels only",
            "periodic": False,
            "axes": [0, 1, 2],
            "percolating_component": (
                "a connected component touching both opposing faces normal to "
                "the evaluated axis"
            ),
            "numerator": "phase-0 pore voxels in all percolating components",
            "denominator": "all phase-0 pore voxels in the volume",
            "reference_mean": reference_percolation,
            "aggregation": ("mean fraction across the three volume axes"),
            "phase_absent": "undefined; evaluation raises ValueError",
            "csv_unit": "fraction",
            "chart_unit": "percent",
        },
        "generation_time": {
            "clock": "time.perf_counter",
            "cuda_synchronized": True,
            "includes": "sampling",
            "excludes": "TIFF serialization and metric calculation",
            "unit": "seconds",
        },
        "tortuosity": {
            "method": "TauFactor steady-state diffusion Solver (classical tortuosity factor)",
            "package_version": version("taufactor"),
            "conductive_phase": PORE_PHASE,
            "conductive_value_supplied_to_taufactor": 1,
            "convergence_criterion": TAUFACTOR_CONVERGENCE,
            "boundary_conditions": "Dirichlet along axis 0; no-flux on transverse boundaries",
            "axis": AXIS,
        },
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_chart(
    rows: list[dict],
    reference_porosity: float,
    reference_tortuosity: float,
    reference_percolation: float,
    output: Path,
) -> None:
    positions = np.asarray([row["anchor_count"] for row in rows], dtype=float)
    figure, panels = plt.subplots(2, 3, figsize=(14.5, 7.3), facecolor="white")
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.12,
        top=0.97,
        hspace=0.30,
        wspace=0.28,
    )
    specs = (
        ("voxel_accuracy", None, "Voxel accuracy (%)", 100.0, None),
        ("fid", None, "FID", 1.0, None),
        (
            "tortuosity_axis0",
            reference_tortuosity,
            "Tortuosity",
            1.0,
            "GT",
        ),
        ("porosity", reference_porosity, "Porosity, phase 0 (%)", 100.0, "GT"),
        (
            "percolation",
            reference_percolation,
            "Percolation, phase 0 (%)",
            100.0,
            "GT",
        ),
        ("generation_seconds", None, "Generation time (s)", 1.0, None),
    )
    for panel, letter, (key, reference, ylabel, scale, reference_label) in zip(
        panels.ravel(),
        "abcdef",
        specs,
        strict=True,
    ):
        values = np.asarray(
            [np.nan if row[key] is None else scale * row[key] for row in rows],
            dtype=float,
        )
        panel.plot(
            positions,
            values,
            color=ROI_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
        )
        if reference is not None and np.isfinite(reference):
            reference_value = scale * reference
            panel.axhline(
                reference_value,
                color="#64748b",
                linewidth=1.2,
                linestyle="--",
                label=f"{reference_label}: {reference_value:.3f}",
            )
            panel.legend(frameon=False, fontsize=9, loc="best")
        panel.set_ylabel(ylabel, fontsize=10)
        maximum = max(COUNTS)
        panel.set_xscale("symlog", base=2, linthresh=1)
        panel.set_xlim(0, maximum)
        panel.set_xticks(tuple(sorted(COUNTS)))
        panel.set_xticklabels(tuple(map(str, sorted(COUNTS))), fontsize=8)
        panel.text(
            0.01,
            0.97,
            f"({letter})",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        panel.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
        panel.spines[["top", "right"]].set_visible(False)
        panel.spines[["left", "bottom"]].set_color("#94a3b8")
        panel.tick_params(colors="#475569")
    panels[0, 0].set_ylim(0, 100)
    figure.supxlabel("Number of supplied axis-0 anchor planes", fontsize=11, y=0.025)
    figure.savefig(
        output,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
